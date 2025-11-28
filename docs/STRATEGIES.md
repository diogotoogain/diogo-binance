# Guia de Estratégias

## FluxoBrabo (Peso: 20%)
Detecta agressão direcional no fluxo de ordens.
- BUY: Fluxo comprador forte
- SELL: Fluxo vendedor forte

## VPINDetector (Peso: 25%)
Detecta traders informados no mercado.
- Alerta quando VPIN alto (risco de movimento)

## OBI - Order Book Imbalance (Peso: 15%)
Analisa desbalanceamento entre bids e asks.
- BUY: Mais pressão compradora
- SELL: Mais pressão vendedora

## RollingVWAP (Peso: 15%)
Detecta sobrecompra/sobrevenda vs VWAP.
- BUY: Preço abaixo da banda inferior
- SELL: Preço acima da banda superior

## PreditorVWAP (Peso: 15%)
Predição de movimento baseada em VWAP.

## LiquidationHunter (Peso: 10%)
Detecta cascatas de liquidação.
