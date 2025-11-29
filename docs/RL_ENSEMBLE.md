# Multi-Timeframe RL Ensemble System

## Visão Geral

O sistema de **CONFLUÊNCIA de múltiplos RLs** treina modelos em diferentes timeframes e usa votação para tomar decisões de trading mais seguras e precisas.

## Conceito

Em vez de usar 1 RL, usamos **3 RLs estudando timeframes diferentes**:

| Timeframe | Nome | Padrão | Episode Length |
|-----------|------|--------|----------------|
| 1m | RL_1m | Scalping | 60 bars (1 hora) |
| 5m | RL_5m | Day Trade | 288 bars (24 horas) |
| 15m | RL_15m | Swing Curto | 192 bars (2 dias) |

A entrada só acontece quando **pelo menos 2 de 3 RLs concordam** (confluência).

## Estrutura de Arquivos

```
v2/
├── src/rl/
│   ├── ensemble.py           # Classe RLEnsemble (votação)
│   ├── ensemble_evaluator.py # Classe EnsembleEvaluator (backtest)
│   └── ...
├── scripts/
│   └── train_multi_timeframe.py  # Script de treino
├── models/rl/
│   ├── ppo_1m_best.zip      # Modelo 1 minuto
│   ├── ppo_5m_best.zip      # Modelo 5 minutos
│   └── ppo_15m_best.zip     # Modelo 15 minutos
└── results/
    ├── latest_results_1m.json
    ├── latest_results_5m.json
    └── latest_results_15m.json
```

## Como Usar

### 1. Treinar os Modelos

```bash
# Treinar todos os timeframes
python v2/scripts/train_multi_timeframe.py

# Treinar timeframe específico
python v2/scripts/train_multi_timeframe.py --timeframe 5m --timesteps 200000

# Opções disponíveis
python v2/scripts/train_multi_timeframe.py --help
```

### 2. Usar o Ensemble para Previsões

```python
from v2.src.rl import RLEnsemble
import numpy as np

# Carregar modelos
ensemble = RLEnsemble(models_dir="v2/models/rl")
ensemble.load_models()

# Verificar status
print(ensemble.get_status())
# {
#     "loaded": True,
#     "total_models": 3,
#     "loaded_timeframes": ["1m", "5m", "15m"],
#     "is_ready": True
# }

# Preparar observações para cada timeframe
observations = {
    "1m": np.array([ofi, tfi, rsi, adx, regime, position, pnl, drawdown]),
    "5m": np.array([ofi, tfi, rsi, adx, regime, position, pnl, drawdown]),
    "15m": np.array([ofi, tfi, rsi, adx, regime, position, pnl, drawdown]),
}

# Obter previsão por votação
result = ensemble.predict(observations)
print(result)
# {
#     "decision": "BUY",
#     "confidence": 2,
#     "confidence_level": "ALTA",
#     "votes": {"1m": "BUY", "5m": "BUY", "15m": "HOLD"},
#     "should_trade": True,
#     "position_size_multiplier": 0.7
# }
```

### 3. Avaliar o Ensemble (Backtest)

```python
from v2.src.rl import RLEnsemble, EnsembleEvaluator
import pandas as pd

# Carregar dados
data = {
    "1m": pd.read_parquet("data/BTCUSDT_1m.parquet"),
    "5m": pd.read_parquet("data/BTCUSDT_5m.parquet"),
    "15m": pd.read_parquet("data/BTCUSDT_15m.parquet"),
}

# Criar ensemble e avaliador
ensemble = RLEnsemble()
ensemble.load_models()

evaluator = EnsembleEvaluator(ensemble)

# Rodar backtest
results = evaluator.backtest_ensemble(data)
print(results["metrics"])
# {
#     "n_trades": 45,
#     "total_return_pct": 12.5,
#     "win_rate": 58.0,
#     "sharpe_ratio": 1.8,
#     "max_drawdown": 8.2
# }

# Comparar com modelos individuais
comparison = evaluator.compare_with_single_model(data)
print(comparison)
#        model  n_trades  win_rate  sharpe_ratio
# 0   ENSEMBLE        45     58.0          1.80
# 1     RL_1m        80     52.0          1.20
# 2     RL_5m        60     55.0          1.45
# 3    RL_15m        35     54.0          1.35
```

## Lógica de Confluência

```python
# A cada 5 minutos:

# 1. Coleta observações de cada timeframe
obs_1m = get_observation_1m()
obs_5m = get_observation_5m()
obs_15m = get_observation_15m()

# 2. Cada RL vota
vote_1m = model_1m.predict(obs_1m)   # "BUY", "SELL", ou "HOLD"
vote_5m = model_5m.predict(obs_5m)
vote_15m = model_15m.predict(obs_15m)

# 3. Conta votos
buy_votes = count_votes("BUY")
sell_votes = count_votes("SELL")

# 4. Decisão por confluência (2 de 3)
if buy_votes >= 2:
    execute_long()
elif sell_votes >= 2:
    execute_short()
else:
    hold()  # Não opera, sem confluência
```

## Níveis de Confiança

| Votos | Nível | Position Size |
|-------|-------|---------------|
| 3/3 | MÁXIMA | 100% |
| 2/3 | ALTA | 70% |
| 1/3 | BAIXA | 0% (não opera) |
| 0/3 | NENHUMA | 0% (não opera) |

## GitHub Actions (Treino Automático)

O workflow `.github/workflows/train-rl-multi-timeframe.yml` treina automaticamente:

- **Schedule**: Todo domingo às 3h UTC
- **Manual**: Via workflow_dispatch

### Executar Manualmente

1. Vá para Actions no GitHub
2. Selecione "Train Multi-Timeframe RL Ensemble"
3. Clique em "Run workflow"
4. Configure os parâmetros:
   - `timesteps`: Número de passos de treino (default: 200000)
   - `timeframe`: Timeframe para treinar (1m, 5m, 15m, ou all)

### Jobs do Workflow

```
train-1m → Treina modelo de 1 minuto (scalping)
     ↓
train-5m → Treina modelo de 5 minutos (day trade)
     ↓
train-15m → Treina modelo de 15 minutos (swing)
     ↓
evaluate-ensemble → Avalia o conjunto (após todos treinarem)
```

## Por que isso é Poderoso

1. **Reduz ruído**: Só opera quando múltiplos timeframes concordam
2. **Mais seguro**: Menos trades, mais qualidade
3. **Diversificação**: Cada RL captura padrões diferentes
4. **Ajustável**: Pode mudar de 2/3 para 3/3 (mais conservador)

## Métricas Esperadas

Após treinamento, espera-se:

- **Win rate do ENSEMBLE** > Win rate individual de cada modelo
- **Menos trades** que modelos individuais (filtro por confluência)
- **Sharpe ratio** maior (menos ruído)
- **Drawdown** menor (decisões mais confiáveis)

## Integração com Trade Executor

O ensemble pode ser integrado com o `TradeExecutor` existente:

```python
from v2.src.rl import RLEnsemble

class TradingBot:
    def __init__(self):
        self.ensemble = RLEnsemble()
        self.ensemble.load_models()
    
    async def check_signal(self):
        # Coleta observações
        observations = await self.get_observations()
        
        # Obtém decisão do ensemble
        result = self.ensemble.predict(observations)
        
        if result["should_trade"]:
            signal = {
                "action": result["decision"],
                "confidence": result["confidence_level"],
                "position_multiplier": result["position_size_multiplier"]
            }
            await self.trade_executor.execute_signal(signal, current_price)
```

## Troubleshooting

### Modelos não carregam
```python
ensemble.load_models()  # Retorna False

# Verifique:
# 1. Os arquivos existem em v2/models/rl/
# 2. stable-baselines3 está instalado
# 3. Os nomes dos arquivos são corretos (ppo_1m_best.zip, etc.)
```

### Erro de dimensão na observação
```python
# Observação deve ter 8 features:
# [ofi, tfi, rsi, adx, regime, position, pnl, drawdown]
obs = np.array([...], dtype=np.float32)
assert obs.shape == (8,) or obs.shape == (1, 8)
```

### Ensemble não pronto
```python
if not ensemble.is_ready():
    # Verifique se pelo menos 2 modelos estão carregados
    print(ensemble.get_loaded_timeframes())
```
