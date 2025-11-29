"""
Schema de Parâmetros Otimizáveis

Define estrutura para ~150 parâmetros que podem ser otimizados via Optuna/RL.
Cada parâmetro tem tipo, range e método de sugestão para integração com Optuna.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union


class ParamType(Enum):
    """Tipos de parâmetros otimizáveis."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    CATEGORICAL = "categorical"


@dataclass
class OptimizableParam:
    """
    Representa um parâmetro otimizável.
    
    Attributes:
        name: Nome do parâmetro (ex: 'risk.max_leverage')
        param_type: Tipo do parâmetro (FLOAT, INT, BOOL, CATEGORICAL)
        default: Valor padrão
        low: Valor mínimo (para FLOAT/INT)
        high: Valor máximo (para FLOAT/INT)
        choices: Lista de valores possíveis (para CATEGORICAL/BOOL)
        step: Passo para discretização (opcional)
        log_scale: Se deve usar escala logarítmica
        description: Descrição do parâmetro
    """
    name: str
    param_type: ParamType
    default: Any
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    step: Optional[Union[float, int]] = None
    log_scale: bool = False
    description: str = ""
    
    def __post_init__(self):
        """Valida parâmetros após inicialização."""
        if self.param_type in (ParamType.FLOAT, ParamType.INT):
            if self.low is None or self.high is None:
                raise ValueError(f"Parâmetro {self.name}: FLOAT/INT requer low e high")
            if self.low > self.high:
                raise ValueError(f"Parâmetro {self.name}: low ({self.low}) > high ({self.high})")
        
        if self.param_type in (ParamType.CATEGORICAL, ParamType.BOOL):
            if self.choices is None or len(self.choices) == 0:
                if self.param_type == ParamType.BOOL:
                    self.choices = [True, False]
                else:
                    raise ValueError(f"Parâmetro {self.name}: CATEGORICAL requer choices")
    
    def to_optuna_suggest(self, trial) -> Any:
        """
        Gera sugestão do Optuna para este parâmetro.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Valor sugerido pelo Optuna
        """
        if self.param_type == ParamType.FLOAT:
            return trial.suggest_float(
                self.name,
                self.low,
                self.high,
                step=self.step,
                log=self.log_scale
            )
        
        elif self.param_type == ParamType.INT:
            return trial.suggest_int(
                self.name,
                self.low,
                self.high,
                step=self.step or 1,
                log=self.log_scale
            )
        
        elif self.param_type == ParamType.BOOL:
            return trial.suggest_categorical(self.name, [True, False])
        
        elif self.param_type == ParamType.CATEGORICAL:
            return trial.suggest_categorical(self.name, self.choices)
        
        raise ValueError(f"Tipo de parâmetro desconhecido: {self.param_type}")
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            'name': self.name,
            'type': self.param_type.value,
            'default': self.default,
            'low': self.low,
            'high': self.high,
            'choices': self.choices,
            'step': self.step,
            'log_scale': self.log_scale,
            'description': self.description
        }


def get_all_optimizable_params() -> List[OptimizableParam]:
    """
    Retorna lista de todos os ~150 parâmetros otimizáveis.
    
    Returns:
        Lista de OptimizableParam
    """
    params = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATA PARAMETERS (~5 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='data.primary_timeframe',
            param_type=ParamType.CATEGORICAL,
            default='1m',
            choices=['1s', '1m', '5m'],
            description='Timeframe primário para análise'
        ),
        OptimizableParam(
            name='data.bar_construction.type',
            param_type=ParamType.CATEGORICAL,
            default='time',
            choices=['time', 'volume', 'dollar'],
            description='Tipo de construção de barras'
        ),
        OptimizableParam(
            name='data.bar_construction.volume_bar_threshold',
            param_type=ParamType.INT,
            default=100,
            low=50,
            high=500,
            description='Threshold para volume bars'
        ),
        OptimizableParam(
            name='data.bar_construction.dollar_bar_threshold',
            param_type=ParamType.INT,
            default=10000000,
            low=5000000,
            high=50000000,
            step=1000000,
            description='Threshold para dollar bars'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # WEBSOCKET PARAMETERS (~3 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='websocket.reconnect_delay',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.5,
            high=5.0,
            description='Delay de reconexão em segundos'
        ),
        OptimizableParam(
            name='websocket.max_reconnect_attempts',
            param_type=ParamType.INT,
            default=10,
            low=5,
            high=20,
            description='Máximo de tentativas de reconexão'
        ),
        OptimizableParam(
            name='websocket.depth_levels',
            param_type=ParamType.CATEGORICAL,
            default=20,
            choices=[5, 10, 20],
            description='Níveis de profundidade do orderbook'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # MICROSTRUCTURE FEATURES (~15 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='features.microstructure.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar features de microestrutura'
        ),
        OptimizableParam(
            name='features.microstructure.ofi.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar OFI'
        ),
        OptimizableParam(
            name='features.microstructure.ofi.window',
            param_type=ParamType.CATEGORICAL,
            default=20,
            choices=[5, 10, 20, 50, 100],
            description='Janela para OFI'
        ),
        OptimizableParam(
            name='features.microstructure.tfi.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar TFI'
        ),
        OptimizableParam(
            name='features.microstructure.tfi.window',
            param_type=ParamType.CATEGORICAL,
            default=20,
            choices=[5, 10, 20, 50, 100],
            description='Janela para TFI'
        ),
        OptimizableParam(
            name='features.microstructure.micro_price.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Micro-Price'
        ),
        OptimizableParam(
            name='features.microstructure.entropy.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Entropia'
        ),
        OptimizableParam(
            name='features.microstructure.entropy.window',
            param_type=ParamType.CATEGORICAL,
            default=50,
            choices=[20, 50, 100, 200],
            description='Janela para Entropia'
        ),
        OptimizableParam(
            name='features.microstructure.vpin.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar VPIN'
        ),
        OptimizableParam(
            name='features.microstructure.vpin.n_buckets',
            param_type=ParamType.CATEGORICAL,
            default=50,
            choices=[20, 50, 100],
            description='Número de buckets para VPIN'
        ),
        OptimizableParam(
            name='features.microstructure.vpin.window',
            param_type=ParamType.CATEGORICAL,
            default=50,
            choices=[20, 50, 100],
            description='Janela para VPIN'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TECHNICAL FEATURES (~25 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='features.technical.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar features técnicas'
        ),
        OptimizableParam(
            name='features.technical.ema.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar EMAs'
        ),
        OptimizableParam(
            name='features.technical.rsi.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar RSI'
        ),
        OptimizableParam(
            name='features.technical.rsi.period',
            param_type=ParamType.CATEGORICAL,
            default=14,
            choices=[7, 14, 21, 28],
            description='Período do RSI'
        ),
        OptimizableParam(
            name='features.technical.rsi.overbought',
            param_type=ParamType.INT,
            default=70,
            low=65,
            high=80,
            description='Nível de sobrecompra RSI'
        ),
        OptimizableParam(
            name='features.technical.rsi.oversold',
            param_type=ParamType.INT,
            default=30,
            low=20,
            high=35,
            description='Nível de sobrevenda RSI'
        ),
        OptimizableParam(
            name='features.technical.macd.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar MACD'
        ),
        OptimizableParam(
            name='features.technical.macd.fast',
            param_type=ParamType.CATEGORICAL,
            default=12,
            choices=[8, 12, 16],
            description='Período rápido MACD'
        ),
        OptimizableParam(
            name='features.technical.macd.slow',
            param_type=ParamType.CATEGORICAL,
            default=26,
            choices=[20, 26, 32],
            description='Período lento MACD'
        ),
        OptimizableParam(
            name='features.technical.macd.signal',
            param_type=ParamType.CATEGORICAL,
            default=9,
            choices=[6, 9, 12],
            description='Período do sinal MACD'
        ),
        OptimizableParam(
            name='features.technical.adx.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar ADX'
        ),
        OptimizableParam(
            name='features.technical.adx.period',
            param_type=ParamType.CATEGORICAL,
            default=14,
            choices=[7, 14, 21],
            description='Período do ADX'
        ),
        OptimizableParam(
            name='features.technical.bollinger.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Bollinger Bands'
        ),
        OptimizableParam(
            name='features.technical.bollinger.period',
            param_type=ParamType.CATEGORICAL,
            default=20,
            choices=[10, 20, 30],
            description='Período das Bollinger Bands'
        ),
        OptimizableParam(
            name='features.technical.bollinger.std_dev',
            param_type=ParamType.FLOAT,
            default=2.0,
            low=1.5,
            high=3.0,
            step=0.1,
            description='Desvio padrão das Bollinger Bands'
        ),
        OptimizableParam(
            name='features.technical.atr.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar ATR'
        ),
        OptimizableParam(
            name='features.technical.atr.period',
            param_type=ParamType.CATEGORICAL,
            default=14,
            choices=[7, 14, 21],
            description='Período do ATR'
        ),
        OptimizableParam(
            name='features.technical.stochastic.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Stochastic'
        ),
        OptimizableParam(
            name='features.technical.stochastic.k_period',
            param_type=ParamType.CATEGORICAL,
            default=14,
            choices=[5, 14, 21],
            description='Período K do Stochastic'
        ),
        OptimizableParam(
            name='features.technical.stochastic.d_period',
            param_type=ParamType.CATEGORICAL,
            default=3,
            choices=[3, 5, 7],
            description='Período D do Stochastic'
        ),
        OptimizableParam(
            name='features.technical.stochastic.overbought',
            param_type=ParamType.INT,
            default=80,
            low=75,
            high=85,
            description='Nível de sobrecompra Stochastic'
        ),
        OptimizableParam(
            name='features.technical.stochastic.oversold',
            param_type=ParamType.INT,
            default=20,
            low=15,
            high=25,
            description='Nível de sobrevenda Stochastic'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME FEATURES (~10 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='features.volume.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar análise de volume'
        ),
        OptimizableParam(
            name='features.volume.volume_spike.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar detecção de volume spike'
        ),
        OptimizableParam(
            name='features.volume.volume_spike.lookback',
            param_type=ParamType.CATEGORICAL,
            default=20,
            choices=[10, 20, 50],
            description='Lookback para volume spike'
        ),
        OptimizableParam(
            name='features.volume.volume_spike.threshold_multiplier',
            param_type=ParamType.FLOAT,
            default=2.0,
            low=1.5,
            high=3.0,
            step=0.1,
            description='Multiplicador threshold para volume spike'
        ),
        OptimizableParam(
            name='features.volume.liquidity_clusters.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar clusters de liquidez'
        ),
        OptimizableParam(
            name='features.volume.liquidity_clusters.levels',
            param_type=ParamType.CATEGORICAL,
            default=10,
            choices=[5, 10, 20],
            description='Número de níveis de clusters'
        ),
        OptimizableParam(
            name='features.volume.liquidity_clusters.threshold_percentile',
            param_type=ParamType.INT,
            default=80,
            low=70,
            high=90,
            description='Percentil threshold para clusters'
        ),
        OptimizableParam(
            name='features.volume.vwap.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar VWAP'
        ),
        OptimizableParam(
            name='features.volume.cvd.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar CVD'
        ),
        OptimizableParam(
            name='features.volume.cvd.window',
            param_type=ParamType.CATEGORICAL,
            default=100,
            choices=[50, 100, 200],
            description='Janela para CVD'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # REGIME DETECTION (~10 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='regime.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar detecção de regime'
        ),
        OptimizableParam(
            name='regime.hmm.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar HMM'
        ),
        OptimizableParam(
            name='regime.hmm.n_regimes',
            param_type=ParamType.CATEGORICAL,
            default=3,
            choices=[2, 3, 4],
            description='Número de regimes HMM'
        ),
        OptimizableParam(
            name='regime.hmm.covariance_type',
            param_type=ParamType.CATEGORICAL,
            default='full',
            choices=['spherical', 'diag', 'full'],
            description='Tipo de covariância HMM'
        ),
        OptimizableParam(
            name='regime.hmm.n_iter',
            param_type=ParamType.INT,
            default=100,
            low=50,
            high=200,
            description='Iterações HMM'
        ),
        OptimizableParam(
            name='regime.adx_based.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar regime baseado em ADX'
        ),
        OptimizableParam(
            name='regime.adx_based.trending_threshold',
            param_type=ParamType.INT,
            default=25,
            low=20,
            high=35,
            description='Threshold ADX para tendência'
        ),
        OptimizableParam(
            name='regime.adx_based.ranging_threshold',
            param_type=ParamType.INT,
            default=20,
            low=15,
            high=25,
            description='Threshold ADX para ranging'
        ),
        OptimizableParam(
            name='regime.volatility_based.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar regime baseado em volatilidade'
        ),
        OptimizableParam(
            name='regime.volatility_based.lookback',
            param_type=ParamType.CATEGORICAL,
            default=100,
            choices=[50, 100, 200],
            description='Lookback para regime de volatilidade'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # LABELING (~8 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='labeling.triple_barrier.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Triple Barrier'
        ),
        OptimizableParam(
            name='labeling.triple_barrier.tp_multiplier',
            param_type=ParamType.FLOAT,
            default=2.0,
            low=1.0,
            high=5.0,
            step=0.5,
            description='Multiplicador TP'
        ),
        OptimizableParam(
            name='labeling.triple_barrier.sl_multiplier',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.5,
            high=2.0,
            step=0.25,
            description='Multiplicador SL'
        ),
        OptimizableParam(
            name='labeling.triple_barrier.max_holding_bars',
            param_type=ParamType.INT,
            default=100,
            low=50,
            high=300,
            description='Máximo de barras de holding'
        ),
        OptimizableParam(
            name='labeling.meta_labeling.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Meta-Labeling'
        ),
        OptimizableParam(
            name='labeling.meta_labeling.model_type',
            param_type=ParamType.CATEGORICAL,
            default='xgboost',
            choices=['xgboost', 'lightgbm', 'rf'],
            description='Tipo de modelo para meta-labeling'
        ),
        OptimizableParam(
            name='labeling.meta_labeling.confidence_threshold',
            param_type=ParamType.FLOAT,
            default=0.6,
            low=0.5,
            high=0.8,
            step=0.05,
            description='Threshold de confiança meta-labeling'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGIES (~40 params)
    # ═══════════════════════════════════════════════════════════════════════
    
    # HFT OFI Scalper
    params.extend([
        OptimizableParam(
            name='strategies.hft_ofi_scalper.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar HFT OFI Scalper'
        ),
        OptimizableParam(
            name='strategies.hft_ofi_scalper.params.ofi_threshold',
            param_type=ParamType.FLOAT,
            default=0.3,
            low=0.1,
            high=0.5,
            step=0.05,
            description='Threshold OFI para entrada'
        ),
        OptimizableParam(
            name='strategies.hft_ofi_scalper.params.tfi_threshold',
            param_type=ParamType.FLOAT,
            default=0.3,
            low=0.1,
            high=0.5,
            step=0.05,
            description='Threshold TFI para entrada'
        ),
        OptimizableParam(
            name='strategies.hft_ofi_scalper.params.min_spread_bps',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.5,
            high=3.0,
            step=0.25,
            description='Spread mínimo em bps'
        ),
        OptimizableParam(
            name='strategies.hft_ofi_scalper.params.holding_seconds',
            param_type=ParamType.INT,
            default=30,
            low=10,
            high=120,
            description='Tempo de holding em segundos'
        ),
        OptimizableParam(
            name='strategies.hft_ofi_scalper.filters.adx_filter.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar filtro ADX'
        ),
        OptimizableParam(
            name='strategies.hft_ofi_scalper.filters.adx_filter.max_adx',
            param_type=ParamType.INT,
            default=30,
            low=20,
            high=40,
            description='ADX máximo para scalping'
        ),
        OptimizableParam(
            name='strategies.hft_ofi_scalper.throttling.max_trades_per_minute',
            param_type=ParamType.INT,
            default=5,
            low=1,
            high=10,
            description='Máximo trades por minuto'
        ),
    ])
    
    # Momentum Intraday
    params.extend([
        OptimizableParam(
            name='strategies.momentum_intraday.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Momentum Intraday'
        ),
        OptimizableParam(
            name='strategies.momentum_intraday.params.ema_fast',
            param_type=ParamType.CATEGORICAL,
            default=9,
            choices=[5, 9, 12],
            description='EMA rápida'
        ),
        OptimizableParam(
            name='strategies.momentum_intraday.params.ema_slow',
            param_type=ParamType.CATEGORICAL,
            default=21,
            choices=[15, 21, 30],
            description='EMA lenta'
        ),
        OptimizableParam(
            name='strategies.momentum_intraday.params.rsi_entry_threshold',
            param_type=ParamType.INT,
            default=50,
            low=40,
            high=60,
            description='RSI threshold para entrada'
        ),
        OptimizableParam(
            name='strategies.momentum_intraday.params.holding_minutes',
            param_type=ParamType.INT,
            default=60,
            low=30,
            high=240,
            description='Tempo de holding em minutos'
        ),
        OptimizableParam(
            name='strategies.momentum_intraday.filters.adx_filter.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar filtro ADX'
        ),
        OptimizableParam(
            name='strategies.momentum_intraday.filters.adx_filter.min_adx',
            param_type=ParamType.INT,
            default=25,
            low=20,
            high=35,
            description='ADX mínimo para momentum'
        ),
        OptimizableParam(
            name='strategies.momentum_intraday.throttling.max_trades_per_hour',
            param_type=ParamType.INT,
            default=4,
            low=1,
            high=10,
            description='Máximo trades por hora'
        ),
    ])
    
    # Mean Reversion Intraday
    params.extend([
        OptimizableParam(
            name='strategies.mean_reversion_intraday.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Mean Reversion'
        ),
        OptimizableParam(
            name='strategies.mean_reversion_intraday.params.zscore_entry',
            param_type=ParamType.FLOAT,
            default=2.0,
            low=1.5,
            high=3.0,
            step=0.25,
            description='Z-score para entrada'
        ),
        OptimizableParam(
            name='strategies.mean_reversion_intraday.params.zscore_exit',
            param_type=ParamType.FLOAT,
            default=0.5,
            low=0.0,
            high=1.0,
            step=0.25,
            description='Z-score para saída'
        ),
        OptimizableParam(
            name='strategies.mean_reversion_intraday.params.bollinger_entry_std',
            param_type=ParamType.FLOAT,
            default=2.0,
            low=1.5,
            high=3.0,
            step=0.25,
            description='Desvio padrão Bollinger para entrada'
        ),
        OptimizableParam(
            name='strategies.mean_reversion_intraday.params.holding_minutes',
            param_type=ParamType.INT,
            default=30,
            low=15,
            high=120,
            description='Tempo de holding em minutos'
        ),
        OptimizableParam(
            name='strategies.mean_reversion_intraday.filters.adx_filter.max_adx',
            param_type=ParamType.INT,
            default=20,
            low=15,
            high=30,
            description='ADX máximo para mean reversion'
        ),
        OptimizableParam(
            name='strategies.mean_reversion_intraday.throttling.max_trades_per_hour',
            param_type=ParamType.INT,
            default=6,
            low=2,
            high=12,
            description='Máximo trades por hora'
        ),
    ])
    
    # Volatility Breakout
    params.extend([
        OptimizableParam(
            name='strategies.volatility_breakout.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Volatility Breakout'
        ),
        OptimizableParam(
            name='strategies.volatility_breakout.params.squeeze_bb_width_percentile',
            param_type=ParamType.INT,
            default=20,
            low=10,
            high=30,
            description='Percentil de squeeze BB'
        ),
        OptimizableParam(
            name='strategies.volatility_breakout.params.breakout_atr_multiplier',
            param_type=ParamType.FLOAT,
            default=1.5,
            low=1.0,
            high=3.0,
            step=0.25,
            description='Multiplicador ATR para breakout'
        ),
        OptimizableParam(
            name='strategies.volatility_breakout.params.volume_confirmation_mult',
            param_type=ParamType.FLOAT,
            default=1.5,
            low=1.2,
            high=2.5,
            step=0.1,
            description='Multiplicador de confirmação de volume'
        ),
        OptimizableParam(
            name='strategies.volatility_breakout.params.holding_minutes',
            param_type=ParamType.INT,
            default=120,
            low=60,
            high=480,
            description='Tempo de holding em minutos'
        ),
        OptimizableParam(
            name='strategies.volatility_breakout.throttling.max_trades_per_day',
            param_type=ParamType.INT,
            default=3,
            low=1,
            high=6,
            description='Máximo trades por dia'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # ENSEMBLE (~10 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='ensemble.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar ensemble'
        ),
        OptimizableParam(
            name='ensemble.weighting_method',
            param_type=ParamType.CATEGORICAL,
            default='sharpe',
            choices=['equal', 'sharpe', 'sortino', 'calmar'],
            description='Método de ponderação'
        ),
        OptimizableParam(
            name='ensemble.lookback_days',
            param_type=ParamType.CATEGORICAL,
            default=30,
            choices=[7, 14, 30, 60],
            description='Lookback em dias'
        ),
        OptimizableParam(
            name='ensemble.min_confidence',
            param_type=ParamType.FLOAT,
            default=0.5,
            low=0.3,
            high=0.7,
            step=0.05,
            description='Confiança mínima'
        ),
        OptimizableParam(
            name='ensemble.rebalance_frequency_hours',
            param_type=ParamType.CATEGORICAL,
            default=24,
            choices=[6, 12, 24, 48],
            description='Frequência de rebalanceamento'
        ),
        OptimizableParam(
            name='ensemble.strategy_constraints.max_weight_per_strategy',
            param_type=ParamType.FLOAT,
            default=0.6,
            low=0.3,
            high=0.8,
            step=0.1,
            description='Peso máximo por estratégia'
        ),
        OptimizableParam(
            name='ensemble.disable_strategy_if.max_drawdown_pct',
            param_type=ParamType.FLOAT,
            default=15.0,
            low=10.0,
            high=25.0,
            step=1.0,
            description='Drawdown máximo para desativar'
        ),
        OptimizableParam(
            name='ensemble.disable_strategy_if.min_sharpe',
            param_type=ParamType.FLOAT,
            default=0.5,
            low=0.0,
            high=1.0,
            step=0.1,
            description='Sharpe mínimo para manter'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # RISK MANAGEMENT (~15 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='risk.risk_per_trade_pct',
            param_type=ParamType.CATEGORICAL,
            default=1.0,
            choices=[0.25, 0.5, 1.0, 1.5, 2.0],
            description='Risco por trade em %'
        ),
        OptimizableParam(
            name='risk.max_daily_loss_pct',
            param_type=ParamType.FLOAT,
            default=3.0,
            low=1.0,
            high=5.0,
            step=0.5,
            description='Perda máxima diária em %'
        ),
        OptimizableParam(
            name='risk.max_weekly_loss_pct',
            param_type=ParamType.FLOAT,
            default=7.0,
            low=3.0,
            high=10.0,
            step=0.5,
            description='Perda máxima semanal em %'
        ),
        OptimizableParam(
            name='risk.max_drawdown_pct',
            param_type=ParamType.FLOAT,
            default=20.0,
            low=10.0,
            high=30.0,
            step=2.5,
            description='Drawdown máximo em %'
        ),
        OptimizableParam(
            name='risk.max_open_positions',
            param_type=ParamType.CATEGORICAL,
            default=1,
            choices=[1, 2, 3],
            description='Posições abertas máximas'
        ),
        OptimizableParam(
            name='risk.max_leverage',
            param_type=ParamType.CATEGORICAL,
            default=10,
            choices=[3, 5, 10, 15, 20],
            description='Alavancagem máxima'
        ),
        OptimizableParam(
            name='risk.regime_risk_adjustment.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar ajuste por regime'
        ),
        OptimizableParam(
            name='risk.regime_risk_adjustment.crash_regime_multiplier',
            param_type=ParamType.FLOAT,
            default=0.3,
            low=0.1,
            high=0.5,
            step=0.05,
            description='Multiplicador regime crash'
        ),
        OptimizableParam(
            name='risk.regime_risk_adjustment.high_vol_regime_multiplier',
            param_type=ParamType.FLOAT,
            default=0.5,
            low=0.3,
            high=0.7,
            step=0.05,
            description='Multiplicador regime alta vol'
        ),
        OptimizableParam(
            name='risk.kill_switch.max_loss_trigger_pct',
            param_type=ParamType.FLOAT,
            default=5.0,
            low=3.0,
            high=10.0,
            step=0.5,
            description='Trigger de perda do kill switch'
        ),
        OptimizableParam(
            name='risk.kill_switch.pause_duration_hours',
            param_type=ParamType.INT,
            default=24,
            low=12,
            high=48,
            description='Duração de pausa do kill switch'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT (~12 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='position.sl.default_pct',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.5,
            high=3.0,
            step=0.25,
            description='Stop loss padrão em %'
        ),
        OptimizableParam(
            name='position.sl.atr_multiplier',
            param_type=ParamType.FLOAT,
            default=2.0,
            low=1.0,
            high=4.0,
            step=0.5,
            description='Multiplicador ATR para SL'
        ),
        OptimizableParam(
            name='position.sl.use_atr_based',
            param_type=ParamType.BOOL,
            default=True,
            description='Usar SL baseado em ATR'
        ),
        OptimizableParam(
            name='position.tp.default_pct',
            param_type=ParamType.FLOAT,
            default=2.0,
            low=1.0,
            high=5.0,
            step=0.5,
            description='Take profit padrão em %'
        ),
        OptimizableParam(
            name='position.tp.atr_multiplier',
            param_type=ParamType.FLOAT,
            default=3.0,
            low=1.5,
            high=6.0,
            step=0.5,
            description='Multiplicador ATR para TP'
        ),
        OptimizableParam(
            name='position.tp.risk_reward_ratio',
            param_type=ParamType.FLOAT,
            default=2.0,
            low=1.0,
            high=4.0,
            step=0.5,
            description='Ratio risco/retorno'
        ),
        OptimizableParam(
            name='position.trailing.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar trailing stop'
        ),
        OptimizableParam(
            name='position.trailing.activation_pct',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.5,
            high=2.0,
            step=0.25,
            description='Ativação do trailing em %'
        ),
        OptimizableParam(
            name='position.trailing.trailing_pct',
            param_type=ParamType.FLOAT,
            default=0.5,
            low=0.2,
            high=1.0,
            step=0.1,
            description='Trailing em %'
        ),
        OptimizableParam(
            name='position.partial_exits.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar saídas parciais'
        ),
        OptimizableParam(
            name='position.time_based_exit.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar saída por tempo'
        ),
        OptimizableParam(
            name='position.time_based_exit.max_holding_hours',
            param_type=ParamType.INT,
            default=24,
            low=4,
            high=72,
            description='Tempo máximo de holding em horas'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # BET SIZING (~10 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='bet_sizing.method',
            param_type=ParamType.CATEGORICAL,
            default='kelly',
            choices=['fixed', 'kelly', 'vol_target', 'rl'],
            description='Método de sizing'
        ),
        OptimizableParam(
            name='bet_sizing.fixed.position_size_pct',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.5,
            high=3.0,
            step=0.25,
            description='Tamanho fixo de posição em %'
        ),
        OptimizableParam(
            name='bet_sizing.kelly.fraction',
            param_type=ParamType.FLOAT,
            default=0.25,
            low=0.1,
            high=0.5,
            step=0.05,
            description='Fração Kelly'
        ),
        OptimizableParam(
            name='bet_sizing.kelly.max_kelly',
            param_type=ParamType.FLOAT,
            default=0.5,
            low=0.25,
            high=0.75,
            step=0.05,
            description='Kelly máximo'
        ),
        OptimizableParam(
            name='bet_sizing.kelly.lookback_trades',
            param_type=ParamType.INT,
            default=100,
            low=50,
            high=200,
            description='Lookback trades para Kelly'
        ),
        OptimizableParam(
            name='bet_sizing.vol_target.target_annual_vol',
            param_type=ParamType.FLOAT,
            default=0.15,
            low=0.1,
            high=0.3,
            step=0.025,
            description='Vol target anualizado'
        ),
        OptimizableParam(
            name='bet_sizing.vol_target.vol_lookback',
            param_type=ParamType.CATEGORICAL,
            default=20,
            choices=[10, 20, 50],
            description='Lookback para vol'
        ),
        OptimizableParam(
            name='bet_sizing.rl.enabled',
            param_type=ParamType.BOOL,
            default=False,
            description='Habilitar RL sizing'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # RL (~15 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='rl.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar RL'
        ),
        OptimizableParam(
            name='rl.environment.lookback_bars',
            param_type=ParamType.INT,
            default=50,
            low=20,
            high=100,
            description='Lookback bars para observação'
        ),
        OptimizableParam(
            name='rl.environment.reward_scaling',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.1,
            high=10.0,
            log_scale=True,
            description='Scaling de reward'
        ),
        OptimizableParam(
            name='rl.action_space.type',
            param_type=ParamType.CATEGORICAL,
            default='discrete',
            choices=['discrete', 'continuous'],
            description='Tipo de action space'
        ),
        OptimizableParam(
            name='rl.agents.ppo.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar PPO'
        ),
        OptimizableParam(
            name='rl.agents.ppo.learning_rate',
            param_type=ParamType.FLOAT,
            default=0.0003,
            low=0.00001,
            high=0.001,
            log_scale=True,
            description='Learning rate PPO'
        ),
        OptimizableParam(
            name='rl.agents.ppo.n_steps',
            param_type=ParamType.CATEGORICAL,
            default=2048,
            choices=[1024, 2048, 4096],
            description='N steps PPO'
        ),
        OptimizableParam(
            name='rl.agents.ppo.batch_size',
            param_type=ParamType.CATEGORICAL,
            default=64,
            choices=[32, 64, 128],
            description='Batch size PPO'
        ),
        OptimizableParam(
            name='rl.agents.ppo.gamma',
            param_type=ParamType.FLOAT,
            default=0.99,
            low=0.95,
            high=0.999,
            description='Gamma PPO'
        ),
        OptimizableParam(
            name='rl.agents.sac.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar SAC'
        ),
        OptimizableParam(
            name='rl.agents.sac.learning_rate',
            param_type=ParamType.FLOAT,
            default=0.0003,
            low=0.00001,
            high=0.001,
            log_scale=True,
            description='Learning rate SAC'
        ),
        OptimizableParam(
            name='rl.agents.sac.gamma',
            param_type=ParamType.FLOAT,
            default=0.99,
            low=0.95,
            high=0.999,
            description='Gamma SAC'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # ONLINE LEARNING (~8 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='online_learning.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar online learning'
        ),
        OptimizableParam(
            name='online_learning.river_models.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar River models'
        ),
        OptimizableParam(
            name='online_learning.river_models.model_type',
            param_type=ParamType.CATEGORICAL,
            default='HoeffdingTree',
            choices=['HoeffdingTree', 'ARFF', 'LinearRegression'],
            description='Tipo de modelo River'
        ),
        OptimizableParam(
            name='online_learning.drift_detection.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar detecção de drift'
        ),
        OptimizableParam(
            name='online_learning.drift_detection.method',
            param_type=ParamType.CATEGORICAL,
            default='ADWIN',
            choices=['ADWIN', 'DDM', 'EDDM'],
            description='Método de detecção de drift'
        ),
        OptimizableParam(
            name='online_learning.drift_detection.delta',
            param_type=ParamType.FLOAT,
            default=0.002,
            low=0.001,
            high=0.01,
            log_scale=True,
            description='Delta para detecção de drift'
        ),
        OptimizableParam(
            name='online_learning.drift_detection.retrain_on_drift',
            param_type=ParamType.BOOL,
            default=True,
            description='Retreinar ao detectar drift'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # OPTIMIZATION (~8 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='optimization.objectives.primary',
            param_type=ParamType.CATEGORICAL,
            default='sharpe',
            choices=['sharpe', 'sortino', 'calmar'],
            description='Objetivo primário'
        ),
        OptimizableParam(
            name='optimization.pruning.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar pruning'
        ),
        OptimizableParam(
            name='optimization.pruning.method',
            param_type=ParamType.CATEGORICAL,
            default='median',
            choices=['median', 'percentile', 'hyperband'],
            description='Método de pruning'
        ),
        OptimizableParam(
            name='optimization.sampler.type',
            param_type=ParamType.CATEGORICAL,
            default='TPE',
            choices=['TPE', 'CMA-ES', 'Random'],
            description='Tipo de sampler'
        ),
        OptimizableParam(
            name='optimization.feature_selection.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar feature selection'
        ),
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # BACKTEST (~10 params)
    # ═══════════════════════════════════════════════════════════════════════
    params.extend([
        OptimizableParam(
            name='backtest.slippage_model.type',
            param_type=ParamType.CATEGORICAL,
            default='volume_based',
            choices=['fixed', 'volume_based', 'spread_based'],
            description='Tipo de modelo de slippage'
        ),
        OptimizableParam(
            name='backtest.slippage_model.base_slippage_bps',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.5,
            high=3.0,
            step=0.25,
            description='Slippage base em bps'
        ),
        OptimizableParam(
            name='backtest.slippage_model.volume_impact_factor',
            param_type=ParamType.FLOAT,
            default=0.5,
            low=0.1,
            high=1.0,
            step=0.1,
            description='Fator de impacto de volume'
        ),
        OptimizableParam(
            name='backtest.walk_forward.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar walk-forward'
        ),
        OptimizableParam(
            name='backtest.walk_forward.train_window_days',
            param_type=ParamType.CATEGORICAL,
            default=60,
            choices=[30, 60, 90],
            description='Janela de treino em dias'
        ),
        OptimizableParam(
            name='backtest.walk_forward.test_window_days',
            param_type=ParamType.CATEGORICAL,
            default=14,
            choices=[7, 14, 21],
            description='Janela de teste em dias'
        ),
        OptimizableParam(
            name='backtest.monte_carlo.enabled',
            param_type=ParamType.BOOL,
            default=True,
            description='Habilitar Monte Carlo'
        ),
        OptimizableParam(
            name='backtest.monte_carlo.n_simulations',
            param_type=ParamType.INT,
            default=1000,
            low=100,
            high=5000,
            description='Número de simulações Monte Carlo'
        ),
    ])
    
    return params


def count_optimizable_params() -> int:
    """
    Conta o número total de parâmetros otimizáveis.
    
    Returns:
        Número total de parâmetros
    """
    return len(get_all_optimizable_params())


def get_params_by_section(section: str) -> List[OptimizableParam]:
    """
    Retorna parâmetros de uma seção específica.
    
    Args:
        section: Nome da seção (ex: 'risk', 'strategies')
        
    Returns:
        Lista de OptimizableParam da seção
    """
    all_params = get_all_optimizable_params()
    return [p for p in all_params if p.name.startswith(section)]
