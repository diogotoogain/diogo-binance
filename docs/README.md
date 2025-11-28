# SNAME-MR Trading Bot - Documentação

## Visão Geral

Sistema de trading automatizado para Binance Futures usando múltiplas estratégias de microestrutura de mercado.

## Arquitetura

O sistema usa:
- BinanceConnector: Conexão WebSocket e REST
- EventBus: Comunicação pub/sub entre módulos
- Estratégias: FluxoBrabo, VPIN, OBI, RollingVWAP, PreditorVWAP
- MetaController: Combina sinais das estratégias
- TradeExecutor: Executa ordens na Binance
- RiskManager: Controla risco por trade e diário
- PositionManager: Gerencia posições, SL, TP

## Fluxo de Dados

1. BinanceConnector recebe ticks via WebSocket
2. EventBus distribui para todas as estratégias
3. Estratégias analisam e emitem sinais
4. MetaController combina sinais (votação ponderada)
5. Se houver consenso, TradeExecutor executa
6. RiskManager valida antes de cada trade
