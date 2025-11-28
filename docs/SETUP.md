# Setup do Sistema

## Requisitos

- Python 3.10+
- Conta Binance com API Demo

## Instalação

1. Clone o repositório
2. Crie ambiente virtual: `python -m venv venv`
3. Ative: `source venv/bin/activate`
4. Instale: `pip install -r requirements.txt`
5. Configure o `.env` com suas chaves
6. Execute: `python main.py`

## Variáveis de Ambiente Principais

- `BINANCE_API_KEY`: Chave API
- `BINANCE_SECRET_KEY`: Secret
- `USE_DEMO`: true para ambiente demo
- `SYMBOL`: Par de trading (BTCUSDT)
- `RISK_PER_TRADE`: Risco por trade (0.01 = 1%)
- `MAX_DAILY_LOSS`: Stop diário (0.03 = 3%)
- `MIN_STRATEGIES_AGREE`: Mínimo de estratégias para trade
- `MIN_CONFIDENCE`: Confiança mínima (HIGH)
