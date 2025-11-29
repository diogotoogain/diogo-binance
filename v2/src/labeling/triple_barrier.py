"""
Triple Barrier Method for Trade Labeling.

Implementation of Marcos López de Prado's Triple Barrier Method.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class TripleBarrierLabeler:
    """
    Triple Barrier Method para labeling de trades.
    
    Parâmetros do config (v2/config/default.yaml):
    - labeling.triple_barrier.tp_multiplier: 2.0    # OPTIMIZE: [1.0, 1.5, 2.0, 2.5, 3.0]
    - labeling.triple_barrier.sl_multiplier: 1.0    # OPTIMIZE: [0.5, 1.0, 1.5, 2.0]
    - labeling.triple_barrier.max_holding_bars: 100 # OPTIMIZE: [50, 100, 200, 500]
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o labeler com parâmetros do config.
        
        Args:
            config: Dicionário de configuração com parâmetros de labeling
        """
        labeling_config = config.get('labeling', {}).get('triple_barrier', {})
        self.tp_mult = labeling_config.get('tp_multiplier', 2.0)
        self.sl_mult = labeling_config.get('sl_multiplier', 1.0)
        self.max_bars = labeling_config.get('max_holding_bars', 100)
        
    def get_barriers(self, price: float, atr: float) -> Tuple[float, float, int]:
        """
        Calcula as 3 barreiras baseadas no preço e ATR.
        
        Args:
            price: Preço atual do ativo
            atr: Average True Range (volatilidade)
            
        Returns:
            Tuple contendo (take_profit, stop_loss, max_bars)
        """
        if price <= 0 or atr <= 0:
            raise ValueError("Price and ATR must be positive values")
            
        tp = price * (1 + self.tp_mult * atr / price)
        sl = price * (1 - self.sl_mult * atr / price)
        return tp, sl, self.max_bars
        
    def label(self, data: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
        """
        Gera labels para eventos usando Triple Barrier Method.
        
        Labels:
            1 = TP hit first (profitable)
            -1 = SL hit first (loss)
            0 = Max holding time reached
            
        Args:
            data: DataFrame com colunas ['close', 'high', 'low', 'atr']
            events: DataFrame com índice de timestamps dos eventos a serem labelados
            
        Returns:
            Series com labels para cada evento
        """
        if data.empty or len(events.index) == 0:
            return pd.Series(dtype=int)
            
        required_columns = {'close', 'high', 'low'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        labels = pd.Series(index=events.index, dtype=int)
        
        for event_idx in events.index:
            if event_idx not in data.index:
                labels.loc[event_idx] = 0
                continue
                
            event_loc = data.index.get_loc(event_idx)
            entry_price = data.loc[event_idx, 'close']
            
            # Usa ATR se disponível, senão calcula uma estimativa
            if 'atr' in data.columns:
                atr = data.loc[event_idx, 'atr']
            else:
                # Estima ATR usando range médio
                lookback = min(14, event_loc)
                if lookback > 0:
                    atr = (data['high'].iloc[event_loc-lookback:event_loc+1] - 
                           data['low'].iloc[event_loc-lookback:event_loc+1]).mean()
                else:
                    atr = entry_price * 0.01  # 1% como fallback
                    
            tp, sl, max_bars = self.get_barriers(entry_price, atr)
            
            # Procura qual barreira é atingida primeiro
            label = self._find_barrier_hit(
                data, event_loc, entry_price, tp, sl, max_bars
            )
            labels.loc[event_idx] = label
            
        return labels
        
    def _find_barrier_hit(
        self, 
        data: pd.DataFrame, 
        start_loc: int, 
        entry_price: float,
        tp: float, 
        sl: float, 
        max_bars: int
    ) -> int:
        """
        Encontra qual barreira foi atingida primeiro.
        
        Args:
            data: DataFrame com dados de preço
            start_loc: Índice de início (localização inteira)
            entry_price: Preço de entrada
            tp: Preço de take profit
            sl: Preço de stop loss
            max_bars: Máximo de barras de holding
            
        Returns:
            Label: 1 (TP), -1 (SL), ou 0 (timeout)
        """
        end_loc = min(start_loc + max_bars + 1, len(data))
        
        for i in range(start_loc + 1, end_loc):
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            
            # Verifica se TP foi atingido (preço subiu até TP)
            tp_hit = high >= tp
            # Verifica se SL foi atingido (preço caiu até SL)
            sl_hit = low <= sl
            
            if tp_hit and sl_hit:
                # Ambos foram atingidos na mesma barra - usa ordem pelo tempo
                # Assume que o mais próximo do open foi atingido primeiro
                if 'open' in data.columns:
                    open_price = data['open'].iloc[i]
                elif i > 0:
                    open_price = data['close'].iloc[i-1]
                else:
                    open_price = entry_price
                if abs(open_price - tp) < abs(open_price - sl):
                    return 1  # TP mais próximo
                else:
                    return -1  # SL mais próximo
            elif tp_hit:
                return 1
            elif sl_hit:
                return -1
                
        return 0  # Max holding time reached
        
    def get_barrier_info(self, price: float, atr: float) -> Dict:
        """
        Retorna informações detalhadas sobre as barreiras.
        
        Args:
            price: Preço atual
            atr: Average True Range
            
        Returns:
            Dicionário com informações das barreiras
        """
        tp, sl, max_bars = self.get_barriers(price, atr)
        
        return {
            'entry_price': price,
            'take_profit': tp,
            'stop_loss': sl,
            'max_holding_bars': max_bars,
            'tp_distance_pct': (tp - price) / price * 100,
            'sl_distance_pct': (price - sl) / price * 100,
            'risk_reward_ratio': (tp - price) / (price - sl) if price > sl else float('inf')
        }
