# Configuração Avançada

## Pesos das Estratégias (no .env)

```
WEIGHT_FLUXO=0.20
WEIGHT_VWAP=0.15
WEIGHT_PREDITOR=0.15
WEIGHT_VPIN=0.25
WEIGHT_OBI=0.15
WEIGHT_LIQ=0.10
```

## Meta-Controlador

```
MIN_STRATEGIES_AGREE=2     # mínimo de estratégias concordando
SIGNAL_TIMEOUT=5.0         # segundos para expirar sinal
DEBOUNCE_SECONDS=3.0       # cooldown entre execuções
```

## Risk Management

```
RISK_PER_TRADE=0.01        # 1% do saldo por trade
MAX_POSITION_SIZE=0.05     # máximo 5% em uma posição
MAX_DAILY_LOSS=0.03        # stop diário de 3%
MAX_CONCURRENT_POSITIONS=1
```

## Position Management

```
DEFAULT_SL_PERCENT=0.01    # Stop Loss 1%
DEFAULT_TP_PERCENT=0.02    # Take Profit 2%
USE_TRAILING_STOP=false
TRAILING_PERCENT=0.005
```
