import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def calcular_lucro_simulado():
    print("\n--- 💰 RELATÓRIO FINANCEIRO (SIMULAÇÃO 100% MARGIN) ---")
    
    # Conecta no Banco
    db_url = os.getenv("DATABASE_URL")
    if db_url and "+asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "")
        
    engine = create_engine(db_url)
    
    # Pega todos os sinais ordenados
    try:
        query = "SELECT * FROM signals ORDER BY timestamp ASC"
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Erro ao ler banco: {e}")
        return
    
    if df.empty:
        print("⚠️ Nenhum sinal para calcular.")
        return

    # --- PARÂMETROS INICIAIS ---
    saldo_inicial = 5000.00 # USDT
    saldo_atual = saldo_inicial
    posicao_btc = 0.0 # Quantos BTCs temos
    preco_entrada = 0.0
    taxa_corretagem = 0.0004 # 0.04% (Taker Binance Futures)
    
    trades_realizados = 0
    vitorias = 0
    
    print(f"💵 Saldo Inicial: ${saldo_inicial:.2f}")
    print("-" * 60)

    for index, row in df.iterrows():
        # Extrai o preço do sinal
        meta = row['metadata_info']
        # Garante que é um dicionário antes de tentar acessar
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except:
                continue
                
        preco_sinal = float(meta.get('price', 0))
        
        if preco_sinal == 0: continue # Ignora sinal sem preço

        tipo_ordem = row['signal_type'] # BUY ou SELL
        
        # LÓGICA DE EXECUÇÃO (VIRA-MÃO)
        
        # 1. Se veio COMPRA
        if tipo_ordem == 'BUY':
            # Se estou VENDIDO (Short), preciso comprar para zerar
            if posicao_btc < 0:
                lucro_trade = (preco_entrada - preco_sinal) * abs(posicao_btc)
                custo_taxa = (preco_sinal * abs(posicao_btc)) * taxa_corretagem
                saldo_atual += lucro_trade - custo_taxa
                
                if lucro_trade > 0: vitorias += 1
                trades_realizados += 1
                posicao_btc = 0 # Zerado
                
            # Se estou NEUTRO (ou zerei agora), entro COMPRADO (Long)
            if posicao_btc == 0:
                # Compra o máximo possível com o saldo atual
                posicao_btc = saldo_atual / preco_sinal
                preco_entrada = preco_sinal
                # Paga taxa de entrada
                saldo_atual -= (saldo_atual * taxa_corretagem)

        # 2. Se veio VENDA
        elif tipo_ordem == 'SELL':
            # Se estou COMPRADO (Long), preciso vender para zerar
            if posicao_btc > 0:
                lucro_trade = (preco_sinal - preco_entrada) * abs(posicao_btc)
                custo_taxa = (preco_sinal * abs(posicao_btc)) * taxa_corretagem
                saldo_atual += lucro_trade - custo_taxa
                
                if lucro_trade > 0: vitorias += 1
                trades_realizados += 1
                posicao_btc = 0 # Zerado
                
            # Se estou NEUTRO, entro VENDIDO (Short)
            if posicao_btc == 0:
                # Vende a descoberto
                posicao_btc = - (saldo_atual / preco_sinal)
                preco_entrada = preco_sinal
                # Paga taxa de entrada
                saldo_atual -= (saldo_atual * taxa_corretagem)

    # --- RESULTADO FINAL ---
    # Se terminou posicionado, marca a mercado com o último preço
    if not df.empty:
        ultimo_meta = df.iloc[-1]['metadata_info']
        if isinstance(ultimo_meta, str):
             import json
             ultimo_meta = json.loads(ultimo_meta)
             
        ultimo_preco = float(ultimo_meta.get('price', 0))
    
        if posicao_btc != 0 and ultimo_preco > 0:
            print(f"⚠️ Posição Aberta no final: {posicao_btc:.4f} BTC")
            if posicao_btc > 0: # Long
                flutuante = (ultimo_preco - preco_entrada) * posicao_btc
            else: # Short
                flutuante = (preco_entrada - ultimo_preco) * abs(posicao_btc)
            saldo_atual += flutuante
            print(f"   (Fechando posição virtualmente a ${ultimo_preco})")

    lucro_liquido = saldo_atual - saldo_inicial
    porcentagem = (lucro_liquido / saldo_inicial) * 100
    win_rate = (vitorias / trades_realizados * 100) if trades_realizados > 0 else 0

    print("-" * 60)
    print(f"📊 Trades Totais: {trades_realizados}")
    print(f"🎯 Taxa de Acerto: {win_rate:.1f}%")
    print(f"💰 Saldo Final:   ${saldo_atual:.2f}")
    print(f"🚀 Lucro Líquido: ${lucro_liquido:.2f} ({porcentagem:.2f}%)")
    print("-" * 60)

if __name__ == "__main__":
    calcular_lucro_simulado()