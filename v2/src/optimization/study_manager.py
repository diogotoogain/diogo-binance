"""
Study manager for Optuna optimization.

Manages multiple optimization studies, comparison, and persistence.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import optuna

logger = logging.getLogger(__name__)


class StudyManager:
    """
    Manage Optuna optimization studies.

    Handles creation, loading, comparison, and reporting of studies.
    """

    def __init__(
        self,
        storage_path: Union[str, Path] = "v2/data",
        storage_type: str = "sqlite",
    ):
        """
        Initialize study manager.

        Args:
            storage_path: Path for study storage
            storage_type: Storage backend type ('sqlite' or 'memory')
        """
        self.storage_path = Path(storage_path)
        self.storage_type = storage_type
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._studies: Dict[str, optuna.Study] = {}

    def _get_storage_url(self, study_name: str) -> Optional[str]:
        """Get storage URL for a study."""
        if self.storage_type == "sqlite":
            db_path = self.storage_path / f"optuna_{study_name}.db"
            return f"sqlite:///{db_path}"
        return None

    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        load_if_exists: bool = True,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """
        Create or load an optimization study.

        Args:
            study_name: Unique name for the study
            direction: Optimization direction
            load_if_exists: Whether to load existing study
            sampler: Custom sampler (default: TPE)
            pruner: Custom pruner (default: MedianPruner)

        Returns:
            Optuna Study object
        """
        storage = self._get_storage_url(study_name)

        if sampler is None:
            sampler = optuna.samplers.TPESampler(multivariate=True)

        if pruner is None:
            pruner = optuna.pruners.MedianPruner()

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=load_if_exists,
        )

        self._studies[study_name] = study
        logger.info(
            f"Study '{study_name}' created/loaded with {len(study.trials)} trials"
        )

        return study

    def get_study(self, study_name: str) -> Optional[optuna.Study]:
        """
        Get a study by name.

        Args:
            study_name: Name of the study

        Returns:
            Study object or None if not found
        """
        if study_name in self._studies:
            return self._studies[study_name]

        # Try to load from storage
        storage = self._get_storage_url(study_name)
        if storage:
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
                self._studies[study_name] = study
                return study
            except Exception as e:
                logger.warning(f"Could not load study '{study_name}': {e}")

        return None

    def list_studies(self) -> List[str]:
        """
        List all available studies.

        Returns:
            List of study names
        """
        studies = list(self._studies.keys())

        # Also check storage for persisted studies
        if self.storage_type == "sqlite":
            for db_file in self.storage_path.glob("optuna_*.db"):
                study_name = db_file.stem.replace("optuna_", "")
                if study_name not in studies:
                    studies.append(study_name)

        return studies

    def delete_study(self, study_name: str) -> bool:
        """
        Delete a study.

        Args:
            study_name: Name of the study to delete

        Returns:
            True if deleted, False otherwise
        """
        # Remove from cache
        if study_name in self._studies:
            del self._studies[study_name]

        # Delete storage
        if self.storage_type == "sqlite":
            db_path = self.storage_path / f"optuna_{study_name}.db"
            if db_path.exists():
                try:
                    optuna.delete_study(
                        study_name=study_name,
                        storage=self._get_storage_url(study_name),
                    )
                    db_path.unlink()
                    logger.info(f"Deleted study '{study_name}'")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete study '{study_name}': {e}")

        return False

    def get_study_summary(self, study_name: str) -> Dict[str, Any]:
        """
        Get summary information for a study.

        Args:
            study_name: Name of the study

        Returns:
            Summary dictionary
        """
        study = self.get_study(study_name)
        if study is None:
            return {}

        summary = {
            "study_name": study_name,
            "direction": study.direction.name,
            "n_trials": len(study.trials),
            "best_value": None,
            "best_params": None,
            "completed_trials": 0,
            "pruned_trials": 0,
            "failed_trials": 0,
        }

        if len(study.trials) > 0:
            # Count trial states
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    summary["completed_trials"] += 1
                elif trial.state == optuna.trial.TrialState.PRUNED:
                    summary["pruned_trials"] += 1
                elif trial.state == optuna.trial.TrialState.FAIL:
                    summary["failed_trials"] += 1

            # Best trial info
            try:
                summary["best_value"] = study.best_value
                summary["best_params"] = study.best_params
            except ValueError:
                pass  # No completed trials

        return summary

    def compare_studies(
        self, study_names: List[str], metric: str = "best_value"
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple studies.

        Args:
            study_names: List of study names to compare
            metric: Metric to compare

        Returns:
            List of comparison results sorted by metric
        """
        results = []

        for name in study_names:
            summary = self.get_study_summary(name)
            if summary:
                results.append(summary)

        # Sort by metric
        results.sort(
            key=lambda x: x.get(metric, float("-inf")),
            reverse=True,
        )

        return results

    def export_study_results(
        self,
        study_name: str,
        output_path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """
        Export study results to file.

        Args:
            study_name: Name of the study
            output_path: Output file path
            format: Output format ('json', 'csv')
        """
        study = self.get_study(study_name)
        if study is None:
            raise ValueError(f"Study '{study_name}' not found")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = {
                "study_name": study_name,
                "direction": study.direction.name,
                "best_value": study.best_value if len(study.trials) > 0 else None,
                "best_params": study.best_params if len(study.trials) > 0 else None,
                "n_trials": len(study.trials),
                "trials": [],
            }

            for trial in study.trials:
                trial_data = {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": trial.params,
                    "datetime_start": (
                        trial.datetime_start.isoformat()
                        if trial.datetime_start
                        else None
                    ),
                    "datetime_complete": (
                        trial.datetime_complete.isoformat()
                        if trial.datetime_complete
                        else None
                    ),
                }
                data["trials"].append(trial_data)

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            df = study.trials_dataframe()
            df.to_csv(output_path, index=False)

        logger.info(f"Exported study '{study_name}' to {output_path}")

    def create_multi_objective_study(
        self,
        study_name: str,
        directions: List[str],
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """
        Create multi-objective optimization study.

        Args:
            study_name: Name for the study
            directions: List of directions for each objective
            load_if_exists: Whether to load existing study

        Returns:
            Multi-objective Optuna Study
        """
        storage = self._get_storage_url(study_name)

        study = optuna.create_study(
            study_name=study_name,
            directions=directions,
            storage=storage,
            load_if_exists=load_if_exists,
        )

        self._studies[study_name] = study
        logger.info(f"Multi-objective study '{study_name}' created/loaded")

        return study

    def get_pareto_front(
        self, study_name: str
    ) -> List[optuna.trial.FrozenTrial]:
        """
        Get Pareto front trials from multi-objective study.

        Args:
            study_name: Name of the multi-objective study

        Returns:
            List of Pareto-optimal trials
        """
        study = self.get_study(study_name)
        if study is None:
            return []

        try:
            return study.best_trials
        except Exception as e:
            logger.warning(f"Could not get Pareto front: {e}")
            return []

    def generate_report(self, study_name: str) -> str:
        """
        Generate text report for a study.

        Args:
            study_name: Name of the study

        Returns:
            Report string
        """
        summary = self.get_study_summary(study_name)
        if not summary:
            return f"Study '{study_name}' not found"

        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║                    📊 OPTUNA STUDY REPORT                      ║
╠═══════════════════════════════════════════════════════════════╣
║  Study Name:       {summary['study_name']:<40}  ║
║  Direction:        {summary['direction']:<40}  ║
║  Total Trials:     {summary['n_trials']:<40}  ║
║  Completed:        {summary['completed_trials']:<40}  ║
║  Pruned:           {summary['pruned_trials']:<40}  ║
║  Failed:           {summary['failed_trials']:<40}  ║
╠═══════════════════════════════════════════════════════════════╣
║  Best Value:       {str(summary.get('best_value', 'N/A')):<40}  ║
╚═══════════════════════════════════════════════════════════════╝
"""

        if summary.get("best_params"):
            report += "\nBest Parameters:\n"
            for key, value in summary["best_params"].items():
                report += f"  {key}: {value}\n"

        return report
