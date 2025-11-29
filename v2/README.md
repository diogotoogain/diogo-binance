# 🤖 V2 Multi-Strategy Quant Trading Bot

Sistema de trading multi-estratégia com ~150 parâmetros otimizáveis via Optuna/RL.

## 🎯 Filosofia Core

- 🚫 **NENHUM número hardcoded** - Tudo é parâmetro otimizável
- ✅ **TUDO parametrizável** via config YAML
- ✅ **Kill switch SEMPRE ativo** (nunca desativar!)
- ✅ **~150 parâmetros otimizáveis** via Optuna/RL

## 📁 Estrutura

```
v2/
├── config/
│   └── default.yaml              # Config mestre com TODOS parâmetros
├── data/
│   ├── raw/                      # Dados brutos
│   ├── processed/                # Dados processados
│   └── parquet/                  # Logs em Parquet
├── models/
│   ├── strategies/               # Modelos de estratégia
│   └── rl/                       # Modelos RL
├── logs/                         # Arquivos de log
├── src/
│   ├── config/
│   │   ├── loader.py             # Carrega e valida config
│   │   └── schema.py             # Schema de parâmetros otimizáveis
│   ├── connectors/
│   │   ├── binance_client.py     # Cliente REST async
│   │   └── websocket_handler.py  # WebSocket robusto (buffer 10000!)
│   ├── data/
│   │   ├── historical.py         # Download dados históricos
│   │   └── bar_builder.py        # Time/Volume/Dollar bars
│   └── utils/
│       ├── logger.py             # Logging colorido + arquivo
│       └── parquet_manager.py    # Salvar/carregar Parquet
├── scripts/
│   ├── download_historical.py    # CLI para download
│   └── validate_config.py        # CLI para validação
├── tests/
│   ├── test_config.py
│   └── test_binance_client.py
├── requirements.txt
└── README.md
```

## 🚀 Instalação

```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r v2/requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas API keys
```

## ⚙️ Configuração

O arquivo `v2/config/default.yaml` contém TODOS os parâmetros do sistema.

### Parâmetros Principais

```yaml
environment:
  mode: "demo"              # demo | live
  use_demo_header: true     # Header X-MBX-DEMO

risk:
  risk_per_trade_pct: 1.0   # 0.1% - 5.0%
  max_leverage: 10          # 1 - 20
  kill_switch:
    enabled: true           # NUNCA DESATIVAR!

websocket:
  buffer_size: 10000        # MÍNIMO 10000 para evitar overflow!
```

### Validar Configuração

```bash
cd v2
python scripts/validate_config.py

# Contar parâmetros otimizáveis
python scripts/validate_config.py --count-params

# Mostrar todos os parâmetros
python scripts/validate_config.py --show-params
```

## 📥 Download de Dados

```bash
cd v2

# Download básico (BTCUSDT, 6 meses, 1m)
python scripts/download_historical.py

# Download customizado
python scripts/download_historical.py \
    --symbol BTCUSDT \
    --months 12 \
    --timeframes 1m 5m 15m 1h

# Com verbose
python scripts/download_historical.py -v
```

## 🧪 Testes

```bash
cd v2

# Rodar todos os testes
pytest tests/ -v

# Testes específicos
pytest tests/test_config.py -v
pytest tests/test_binance_client.py -v
```

## 🛡️ Kill Switch

O kill switch é uma proteção **OBRIGATÓRIA** que:

- Para todas as operações se perda exceder threshold
- Não pode ser desativado via config
- É validado no loader de configuração
- Erro é gerado se `enabled: false`

```yaml
risk:
  kill_switch:
    enabled: true                 # NUNCA DESATIVAR!
    max_loss_trigger_pct: 5.0     # Trigger de perda
    pause_duration_hours: 24      # Duração da pausa
```

## 📊 Parâmetros Otimizáveis

O sistema possui **~150 parâmetros** otimizáveis via Optuna:

| Seção | Parâmetros |
|-------|------------|
| Features | ~40 |
| Strategies | ~40 |
| Risk | ~15 |
| Ensemble | ~10 |
| Position | ~12 |
| RL | ~15 |
| Others | ~18 |

## 🔗 Referências

- `docs/PROMPT_V3_SPEC.md` - Especificação completa
- `docs/ROADMAP_V2.md` - Roadmap de desenvolvimento
- `.env.example` - Variáveis de ambiente

## 📝 Licença

Uso interno apenas.
