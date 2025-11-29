"""
Shannon Entropy Feature.

Mede a aleatoriedade/previsibilidade do mercado.
Entropia alta = mercado imprevisível
Entropia baixa = mercado previsível
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class ShannonEntropy(Feature):
    """
    Shannon Entropy.
    
    Mede a quantidade de "incerteza" ou "aleatoriedade" nos retornos do mercado.
    
    Parâmetros do config:
        window: Janela de cálculo (default: 50)
        n_bins: Número de bins para discretização (default: 10)
        
    OPTIMIZE: window em [20, 50, 100, 200]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa a Shannon Entropy.
        
        Args:
            config: Deve conter 'window' (tamanho da janela)
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.window = config.get('window', 50)
        self.n_bins = config.get('n_bins', 10)
        
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """
        Calcula entropia de Shannon para um array de valores.
        
        Args:
            values: Array de valores
            
        Returns:
            Valor de entropia (0 = completamente previsível, max = completamente aleatório)
        """
        if len(values) == 0 or np.isnan(values).all():
            return 0.0
            
        # Remove NaN
        values = values[~np.isnan(values)]
        
        if len(values) < 2:
            return 0.0
        
        # Discretiza os valores em bins
        try:
            hist, _ = np.histogram(values, bins=self.n_bins, density=False)
        except ValueError:
            return 0.0
        
        # Normaliza para probabilidades
        total = hist.sum()
        hist = hist / total if total > 0 else hist
        
        # Remove zeros para evitar log(0)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        # Entropia de Shannon: -sum(p * log2(p))
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normaliza para [0, 1] (máximo é log2(n_bins))
        max_entropy = np.log2(self.n_bins)
        if max_entropy > 0:
            entropy = entropy / max_entropy
            
        return entropy
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula a Shannon Entropy para um DataFrame.
        
        Args:
            data: DataFrame com coluna 'close' (para calcular retornos)
            
        Returns:
            pd.Series com valores de entropia normalizados em [0, 1]
        """
        if not self.enabled:
            return pd.Series(index=data.index, dtype=float)
            
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        
        # Calcula retornos logarítmicos
        returns = np.log(data['close'] / data['close'].shift(1))
        
        # Calcula entropia rolling
        entropy_values = []
        for i in range(len(returns)):
            if i < self.window - 1:
                entropy_values.append(np.nan)
            else:
                window_returns = returns.iloc[i - self.window + 1:i + 1].values
                entropy = self._calculate_entropy(window_returns)
                entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=data.index, dtype=float)
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> float:
        """
        Calcula a Shannon Entropy incrementalmente.
        
        Args:
            new_data: Dict com 'close'
            state: Dict com 'returns' (lista de retornos)
            
        Returns:
            Valor atual da entropia em [0, 1]
        """
        if not self.enabled:
            return 0.0
            
        # Inicializa estado se necessário
        if 'returns' not in state:
            state['returns'] = []
            state['prev_close'] = None
        
        close = new_data.get('close', 0.0)
        prev_close = state.get('prev_close')
        
        if prev_close is not None and prev_close > 0 and close > 0:
            log_return = np.log(close / prev_close)
            state['returns'].append(log_return)
            
        state['prev_close'] = close
        
        # Mantém apenas os últimos 'window' retornos
        if len(state['returns']) > self.window:
            state['returns'] = state['returns'][-self.window:]
        
        # Calcula entropia
        if len(state['returns']) < self.window:
            return 0.0
            
        return self._calculate_entropy(np.array(state['returns']))
