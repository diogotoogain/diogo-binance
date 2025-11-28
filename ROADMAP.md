# 🚀 SNAME-MR: ROADMAP DO PROJETO

> **Sistema de Negociação Adaptativo Multi-Estratégia com Meta-Gestão de Risco**
> 
> *"A melhor estratégia do mundo não é UMA estratégia, é um PORTFÓLIO de estratégias descorrelacionadas que transforma você em 'a casa' - o cassino que sempre tem vantagem estatística."*

---

## 📍 ONDE ESTOU AGORA

**Data da última atualização:** 2025-11-28

### ✅ Infraestrutura Base (COMPLETO)
- [x] Arquitetura Event-Driven com EventBus
- [x] Conexão WebSocket Binance (com reconexão automática)
- [x] Stream `@aggTrade` (trades em tempo real)
- [x] Stream `@forceOrder` (liquidações)
- [x] Banco de dados PostgreSQL
- [x] Orquestrador de Estratégias
- [x] Sistema de logging

### ✅ Estratégias Implementadas
- [x] **VPINStrategy** - Detecta fluxo tóxico de traders informados
  - Buckets por volume (não por tempo)
  - Parametrizável: bucket_size, n_buckets, thresholds, cooldown
  - Detecta direção do smart money

### 🔄 Em Progresso (PR #3)
- [ ] **CascadeLiquidationStrategy** - Detecta cascatas de liquidação
- [ ] **FlowImbalanceStrategy** - Desequilíbrio ponderado por USD
- [ ] **RollingVWAPStrategy** - VWAP com janela deslizante
- [ ] **OBIStrategy** - Order Book Imbalance com decaimento exponencial
- [ ] Stream `@depth` - Order Book em tempo real

### ⏳ Próximos Passos
- [ ] Regime Switching (HMM) - Classificar mercado em estados
- [ ] Kelly Fracionário - Position sizing matemático
- [ ] Sistema de Métricas - Para o auto-otimizador
- [ ] Funding Rate Strategy - Sentimento de derivativos
- [ ] Open Interest Analysis - Combustível do mercado

### 🔮 Futuro (Requer mais dados/infra)
- [ ] Auto-Otimizador de Parâmetros
- [ ] PPO + Transformers (Cérebro IA)
- [ ] Avellaneda-Stoikov Market Making
- [ ] HRP (Hierarchical Risk Parity)
- [ ] World Models (GANs) para simulação
- [ ] Backtesting Engine realista

---

## 🎯 PARA ONDE VOU

### A Visão Final: O Cassino

```
┌─────────────────────────────────────────────────────────────┐
│                    🧠 META-CONTROLADOR                       │
│         (PPO + Transformers - Aloca capital entre           │
│          estratégias baseado em regime de mercado)          │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ ESTRATÉGIA A  │    │ ESTRATÉGIA B  │    │ ESTRATÉGIA C  │
│  Sentimento   │    │    Regime     │    │   Liquidez    │
│ Funding + OI  │    │  HMM States   │    │ VPIN + OBI    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  💰 GESTÃO DE RISCO                          │
│     Kelly Fracionário + HRP + Circuit Breakers              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ⚡ EXECUÇÃO                               │
│              Binance API (Paper → Real)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 AS 4 ESTRATÉGIAS DO RELATÓRIO SNAME-MR

### Estratégia A: Sentimento/Derivativos ("O Termostato")
| Componente | Status | Descrição |
|------------|--------|-----------|
| Funding Rate | ⏳ | Taxa de financiamento de futuros perpétuos |
| Open Interest | ⏳ | Volume de contratos abertos |
| Basis (Spot-Futures) | ⏳ | Diferença entre preço spot e futuro |

### Estratégia B: Regime Switching ("O Surfista")
| Componente | Status | Descrição |
|------------|--------|-----------|
| HMM (Hidden Markov Model) | ⏳ | Classificar mercado em estados |
| Estado 0: Consolidação | ⏳ | Baixa volatilidade, mercado lateral |
| Estado 1: Tendência | ⏳ | Alta volatilidade direcional |
| Estado 2: Caos | ⏳ | Volatilidade extrema, crashes/pumps |

### Estratégia C: Microestrutura ("O Caçador")
| Componente | Status | Descrição |
|------------|--------|-----------|
| VPIN | ✅ | Probabilidade de fluxo informado |
| OBI | 🔄 | Desequilíbrio do order book |
| Cascata de Liquidações | 🔄 | Detectar efeito dominó |
| Flow Imbalance | 🔄 | Pressão de compra/venda em USD |

### Estratégia D: Market Making ("O Fazendeiro")
| Componente | Status | Descrição |
|------------|--------|-----------|
| Avellaneda-Stoikov | 🔮 | Modelo de market making ótimo |
| Inventário Dinâmico | 🔮 | Ajustar posição baseado em sentimento |
| Spread Adaptativo | 🔮 | Alargar spread em alta volatilidade |

---

## 🔧 DECISÕES TÉCNICAS

### Stack Atual (Python no Mac)
- **Linguagem:** Python 3.x
- **Async:** asyncio + aiohttp
- **WebSocket:** python-binance
- **Banco:** PostgreSQL
- **Estrutura:** Event-Driven Architecture

### Stack Futuro (Performance)
- **Núcleo:** Rust ou C++ (latência)
- **IA:** PyTorch (PPO + Transformers)
- **Dados:** TimescaleDB ou KDB+
- **Infra:** AWS Tokyo (co-location Binance)

---

## 📈 MÉTRICAS DE SUCESSO

### Para considerar o sistema "pronto para live":
- [ ] Win Rate > 55%
- [ ] Sharpe Ratio > 2.0
- [ ] Max Drawdown < 10%
- [ ] Profit Factor > 1.5
- [ ] 30 dias de paper trading lucrativo

### Fases de Deploy:
1. **Paper Trading** (atual) - Testnet Binance
2. **Live com 1%** - Capital mínimo
3. **Live com 10%** - Validação
4. **Live Full** - Kelly Fracionário ativo

---

## 🧠 PRINCÍPIOS DO PROJETO

1. **PARAMETRIZÁVEL** - Nada hard-coded, tudo ajustável
2. **DESCORRELACIONADO** - Estratégias independentes
3. **ADAPTATIVO** - Se ajusta ao regime de mercado
4. **RESILIENTE** - Circuit breakers, reconexão automática
5. **TRANSPARENTE** - Logs detalhados, métricas em tempo real

---

## 📚 REFERÊNCIAS DO RELATÓRIO

### Papers Acadêmicos Citados:
- Easley, López de Prado, O'Hara (2012) - **VPIN**
- Avellaneda & Stoikov (2008) - **Market Making**
- López de Prado - **HRP (Hierarchical Risk Parity)**
- Kelly (1956) - **Critério de Kelly**

### Conceitos Chave:
- **Microestrutura de Mercado** - Como ordens afetam preços
- **Seleção Adversa** - Trading contra informados
- **Toxicidade de Fluxo** - Detectar smart money
- **Regime Switching** - Mercados mudam de estado

---

## 🚦 LEGENDA DE STATUS

- ✅ **Completo** - Implementado e funcionando
- 🔄 **Em Progresso** - Sendo implementado agora
- ⏳ **Próximo** - Na fila, será feito em breve
- 🔮 **Futuro** - Planejado, precisa de mais infra/dados

---

## 📝 HISTÓRICO DE ATUALIZAÇÕES

| Data | O que foi feito |
|------|-----------------|
| 2025-11-28 | Infraestrutura base completa |
| 2025-11-28 | VPINStrategy implementada (PR #2) |
| 2025-11-28 | 4 novas estratégias em desenvolvimento (PR #3) |

---

> **Lembre-se:** O objetivo não é prever o preço. É ter vantagem estatística em múltiplos cenários, como um cassino que lucra independente de quem ganha cada aposta individual.
