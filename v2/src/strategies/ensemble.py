"""
Ensemble Manager

Combina sinais de múltiplas estratégias com pesos adaptativos.
Suporta vários métodos de ponderação: sharpe, sortino, kelly, equal.
Desabilita automaticamente estratégias com drawdown excessivo.

ZERO hardcoded - todos os parâmetros vêm do config.
"""
from typing import Any, Dict, List, Optional
import logging
import time

from v2.src.strategies.base import Strategy, Signal, SignalDirection


class EnsembleManager:
    """
    Gerenciador de ensemble para combinar sinais de múltiplas estratégias.
    
    Funcionalidades:
    - Ponderação por Sharpe Ratio
    - Ponderação por Sortino Ratio
    - Ponderação por Kelly Criterion
    - Ponderação igual (equal)
    - Desativação automática de estratégias com drawdown excessivo
    - Filtro de confiança mínima
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o Ensemble Manager.
        
        Args:
            config: Configuração do ensemble do YAML
        """
        self.config = config
        self.logger = logging.getLogger("EnsembleManager")
        
        # Configurações
        self.weighting_method = config.get('weighting_method', 'sharpe')
        self.lookback_days = config.get('lookback_days', 30)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.rebalance_frequency_hours = config.get('rebalance_frequency_hours', 24)
        
        # Constraints
        constraints = config.get('strategy_constraints', {})
        self.max_weight_per_strategy = constraints.get('max_weight_per_strategy', 0.6)
        self.min_weight_per_strategy = constraints.get('min_weight_per_strategy', 0.0)
        
        # Disable conditions
        disable_config = config.get('disable_strategy_if', {})
        self.max_drawdown_pct = disable_config.get('max_drawdown_pct', 15.0)
        self.min_sharpe = disable_config.get('min_sharpe', 0.5)
        
        # Estado interno
        self._strategies: Dict[str, Strategy] = {}
        self._weights: Dict[str, float] = {}
        self._disabled_strategies: set = set()
        self._last_rebalance_time: float = 0
        
        self.logger.info(
            f"Ensemble inicializado | Método: {self.weighting_method} | "
            f"Min Conf: {self.min_confidence} | Max DD: {self.max_drawdown_pct}%"
        )
    
    def register_strategy(self, strategy: Strategy) -> None:
        """
        Registra uma estratégia no ensemble.
        
        Args:
            strategy: Instância de Strategy
        """
        self._strategies[strategy.name] = strategy
        self._weights[strategy.name] = 1.0 / len(self._strategies)  # Peso inicial igual
        self.logger.info(f"Estratégia registrada: {strategy.name}")
        
        # Rebalanceia pesos após nova estratégia
        self._rebalance_weights()
    
    def remove_strategy(self, strategy_name: str) -> None:
        """
        Remove uma estratégia do ensemble.
        
        Args:
            strategy_name: Nome da estratégia
        """
        if strategy_name in self._strategies:
            del self._strategies[strategy_name]
            del self._weights[strategy_name]
            self._disabled_strategies.discard(strategy_name)
            self._rebalance_weights()
            self.logger.info(f"Estratégia removida: {strategy_name}")
    
    def _calculate_sharpe_weights(self) -> Dict[str, float]:
        """
        Calcula pesos baseados em Sharpe Ratio.
        
        Returns:
            Dict com pesos por estratégia
        """
        sharpe_ratios = {}
        for name, strategy in self._strategies.items():
            if name not in self._disabled_strategies:
                sharpe = max(0, strategy.get_sharpe_ratio())
                sharpe_ratios[name] = sharpe
        
        total_sharpe = sum(sharpe_ratios.values())
        
        if total_sharpe == 0:
            # Se todas têm Sharpe 0, usa peso igual
            n = len(sharpe_ratios)
            return {name: 1.0 / n for name in sharpe_ratios}
        
        return {name: sharpe / total_sharpe for name, sharpe in sharpe_ratios.items()}
    
    def _calculate_sortino_weights(self) -> Dict[str, float]:
        """
        Calcula pesos baseados em Sortino Ratio.
        
        Returns:
            Dict com pesos por estratégia
        """
        sortino_ratios = {}
        for name, strategy in self._strategies.items():
            if name not in self._disabled_strategies:
                sortino = max(0, strategy.get_sortino_ratio())
                # Trata infinito (sem losses)
                if sortino == float('inf'):
                    sortino = 10.0  # Cap em 10
                sortino_ratios[name] = sortino
        
        total_sortino = sum(sortino_ratios.values())
        
        if total_sortino == 0:
            n = len(sortino_ratios)
            return {name: 1.0 / n for name in sortino_ratios}
        
        return {name: sortino / total_sortino for name, sortino in sortino_ratios.items()}
    
    def _calculate_kelly_weights(self) -> Dict[str, float]:
        """
        Calcula pesos baseados em Kelly Criterion.
        
        Kelly = (p * b - q) / b
        onde:
        - p = probabilidade de ganhar (win rate)
        - q = 1 - p
        - b = odds (média de ganho / média de perda)
        
        Returns:
            Dict com pesos por estratégia
        """
        kelly_fractions = {}
        for name, strategy in self._strategies.items():
            if name not in self._disabled_strategies:
                win_rate = strategy.get_win_rate()
                if win_rate == 0 or win_rate == 1:
                    kelly = 0
                else:
                    # Assumimos odds de 1:1 para simplificar
                    b = 1.0
                    p = win_rate
                    q = 1 - p
                    kelly = (p * b - q) / b
                    kelly = max(0, kelly)  # Não permite kelly negativo
                kelly_fractions[name] = kelly
        
        total_kelly = sum(kelly_fractions.values())
        
        if total_kelly == 0:
            n = len(kelly_fractions)
            return {name: 1.0 / n for name in kelly_fractions}
        
        return {name: kelly / total_kelly for name, kelly in kelly_fractions.items()}
    
    def _calculate_equal_weights(self) -> Dict[str, float]:
        """
        Calcula pesos iguais.
        
        Returns:
            Dict com pesos iguais por estratégia
        """
        active_strategies = [
            name for name in self._strategies.keys()
            if name not in self._disabled_strategies
        ]
        
        if not active_strategies:
            return {}
        
        weight = 1.0 / len(active_strategies)
        return {name: weight for name in active_strategies}
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Aplica constraints de peso min/max.
        
        Args:
            weights: Pesos calculados
        
        Returns:
            Pesos ajustados
        """
        constrained = {}
        
        for name, weight in weights.items():
            constrained[name] = max(
                self.min_weight_per_strategy,
                min(self.max_weight_per_strategy, weight)
            )
        
        # Renormaliza
        total = sum(constrained.values())
        if total > 0:
            return {name: w / total for name, w in constrained.items()}
        return constrained
    
    def _rebalance_weights(self) -> None:
        """Recalcula pesos de todas as estratégias."""
        if self.weighting_method == 'sharpe':
            weights = self._calculate_sharpe_weights()
        elif self.weighting_method == 'sortino':
            weights = self._calculate_sortino_weights()
        elif self.weighting_method == 'kelly':
            weights = self._calculate_kelly_weights()
        else:  # equal
            weights = self._calculate_equal_weights()
        
        self._weights = self._apply_weight_constraints(weights)
        self._last_rebalance_time = time.time()
        
        self.logger.info(f"Pesos rebalanceados: {self._weights}")
    
    def _check_strategy_health(self) -> None:
        """Verifica saúde das estratégias e desabilita se necessário."""
        for name, strategy in self._strategies.items():
            drawdown = strategy.get_current_drawdown() * 100  # Em %
            sharpe = strategy.get_sharpe_ratio()
            
            should_disable = False
            reason = ""
            
            if drawdown > self.max_drawdown_pct:
                should_disable = True
                reason = f"Drawdown {drawdown:.1f}% > {self.max_drawdown_pct}%"
            elif sharpe < self.min_sharpe and strategy._total_signals > 10:
                should_disable = True
                reason = f"Sharpe {sharpe:.2f} < {self.min_sharpe}"
            
            if should_disable and name not in self._disabled_strategies:
                self._disabled_strategies.add(name)
                self.logger.warning(f"⚠️ Estratégia desabilitada: {name} | Motivo: {reason}")
                self._rebalance_weights()
            elif not should_disable and name in self._disabled_strategies:
                self._disabled_strategies.discard(name)
                self.logger.info(f"✅ Estratégia reabilitada: {name}")
                self._rebalance_weights()
    
    def _should_rebalance(self) -> bool:
        """Verifica se deve rebalancear pesos."""
        hours_since_rebalance = (time.time() - self._last_rebalance_time) / 3600
        return hours_since_rebalance >= self.rebalance_frequency_hours
    
    def process_signals(
        self, 
        signals: List[Signal]
    ) -> Optional[Signal]:
        """
        Processa sinais de múltiplas estratégias e combina.
        
        Args:
            signals: Lista de sinais das estratégias
        
        Returns:
            Sinal combinado ou None
        """
        # Verifica saúde das estratégias
        self._check_strategy_health()
        
        # Rebalanceia se necessário
        if self._should_rebalance():
            self._rebalance_weights()
        
        # Filtra sinais
        valid_signals = []
        for signal in signals:
            # Ignora estratégias desabilitadas
            if signal.strategy_name in self._disabled_strategies:
                continue
            
            # Ignora sinais com confiança baixa
            if signal.confidence < self.min_confidence:
                continue
            
            # Ignora sinais HOLD
            if signal.direction == SignalDirection.HOLD:
                continue
            
            valid_signals.append(signal)
        
        if not valid_signals:
            return None
        
        # Agrupa por direção
        buy_signals = [s for s in valid_signals if s.direction == SignalDirection.BUY]
        sell_signals = [s for s in valid_signals if s.direction == SignalDirection.SELL]
        
        # Calcula score ponderado por direção
        buy_score = sum(
            s.confidence * self._weights.get(s.strategy_name, 0)
            for s in buy_signals
        )
        sell_score = sum(
            s.confidence * self._weights.get(s.strategy_name, 0)
            for s in sell_signals
        )
        
        # Decide direção
        if buy_score > sell_score and buy_score > 0:
            direction = SignalDirection.BUY
            winning_signals = buy_signals
            score = buy_score
        elif sell_score > buy_score and sell_score > 0:
            direction = SignalDirection.SELL
            winning_signals = sell_signals
            score = sell_score
        else:
            # Empate ou sem sinais claros
            return None
        
        # Combina metadados
        combined_metadata = {
            'ensemble_score': score,
            'contributing_strategies': [s.strategy_name for s in winning_signals],
            'weights_used': {s.strategy_name: self._weights.get(s.strategy_name, 0) for s in winning_signals},
            'individual_confidences': {s.strategy_name: s.confidence for s in winning_signals},
            'buy_score': buy_score,
            'sell_score': sell_score,
        }
        
        # Calcula confiança combinada (média ponderada)
        total_weight = sum(self._weights.get(s.strategy_name, 0) for s in winning_signals)
        if total_weight > 0:
            combined_confidence = score / total_weight
        else:
            combined_confidence = sum(s.confidence for s in winning_signals) / len(winning_signals)
        
        combined_signal = Signal(
            direction=direction,
            strategy_name="Ensemble",
            confidence=min(1.0, combined_confidence),
            reason=f"Ensemble_{direction.value}_Score={score:.3f}",
            metadata=combined_metadata,
        )
        
        self.logger.info(
            f"🎯 ENSEMBLE: {direction.value} | Score: {score:.3f} | "
            f"Estratégias: {[s.strategy_name for s in winning_signals]}"
        )
        
        return combined_signal
    
    def generate_ensemble_signal(
        self, 
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Gera sinal ensemble processando todas as estratégias.
        
        Args:
            market_data: Dados de mercado para todas as estratégias
        
        Returns:
            Sinal combinado ou None
        """
        signals = []
        
        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue
            if name in self._disabled_strategies:
                continue
            
            signal = strategy.generate_signal(market_data)
            if signal:
                signals.append(signal)
        
        return self.process_signals(signals)
    
    def update_strategy_performance(
        self, 
        strategy_name: str, 
        pnl: float
    ) -> None:
        """
        Atualiza performance de uma estratégia.
        
        Args:
            strategy_name: Nome da estratégia
            pnl: PnL do último trade
        """
        if strategy_name in self._strategies:
            self._strategies[strategy_name].update_performance(pnl)
    
    def get_weights(self) -> Dict[str, float]:
        """Retorna pesos atuais."""
        return self._weights.copy()
    
    def get_disabled_strategies(self) -> List[str]:
        """Retorna lista de estratégias desabilitadas."""
        return list(self._disabled_strategies)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status do ensemble.
        
        Returns:
            Dict com status completo
        """
        strategies_status = {}
        for name, strategy in self._strategies.items():
            status = strategy.get_status()
            status['weight'] = self._weights.get(name, 0)
            status['disabled'] = name in self._disabled_strategies
            strategies_status[name] = status
        
        return {
            'weighting_method': self.weighting_method,
            'min_confidence': self.min_confidence,
            'max_drawdown_pct': self.max_drawdown_pct,
            'min_sharpe': self.min_sharpe,
            'total_strategies': len(self._strategies),
            'active_strategies': len(self._strategies) - len(self._disabled_strategies),
            'disabled_strategies': list(self._disabled_strategies),
            'weights': self._weights,
            'last_rebalance': self._last_rebalance_time,
            'strategies': strategies_status,
        }
