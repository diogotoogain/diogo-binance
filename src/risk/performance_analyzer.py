import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Carrega as configurações
load_dotenv()

def relatorio_performance():
    print("\n--- 📊 RELATÓRIO DE SINAIS (SIMULAÇÃO) ---")
    
    # 1. Ajusta a URL do banco para o Pandas (que não usa async)
    db_url = os.getenv("DATABASE_URL")
    if "+asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "")
    
    try:
        engine = create_engine(db_url)
        
        # 2. Busca os sinais ordenados por tempo
        print("⏳ Buscando histórico no cofre...")
        query = "SELECT * FROM signals ORDER BY timestamp ASC"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("⚠️ Nenhum sinal encontrado no banco. O robô rodou?")
            return

        # 3. Estatísticas Básicas
        total_sinais = len(df)
        compras = len(df[df['signal_type'] == 'BUY'])
        vendas = len(df[df['signal_type'] == 'SELL'])
        
        print(f"\n📈 TOTAL DE SINAIS GERADOS: {total_sinais}")
        print(f"🟢 Sinais de COMPRA: {compras}")
        print(f"🔴 Sinais de VENDA:  {vendas}")
        
        print("\n🔎 ÚLTIMOS 10 SINAIS REGISTRADOS:")
        print("-" * 60)
        # Seleciona colunas relevantes para exibir
        view = df[['timestamp', 'strategy', 'signal_type', 'metadata_info']].tail(10)
        
        for index, row in view.iterrows():
            # Tenta extrair o motivo do metadata (que é um JSON/Dict)
            meta = row['metadata_info']
            motivo = meta.get('reason', 'N/A') if isinstance(meta, dict) else 'N/A'
            preco = meta.get('price', 0) if isinstance(meta, dict) else 0
            
            print(f"{row['timestamp']} | {row['strategy']:<15} | {row['signal_type']} | $ {preco} | {motivo}")
            
        print("-" * 60)
        print("✅ Relatório concluído.")

    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {e}")

if __name__ == "__main__":
    relatorio_performance()