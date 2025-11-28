import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def exportar_para_csv():
    print("--- 📄 INICIANDO EXPORTAÇÃO PARA EXCEL/CSV ---")
    
    # 1. Pega a senha do cofre
    db_url = os.getenv("DATABASE_URL")
    
    # Ajuste técnico: O Pandas prefere o driver padrão, não o async
    if "+asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "")
    
    try:
        engine = create_engine(db_url)
        
        print("1. Conectando ao Banco de Dados...")
        # 2. Lê a tabela inteira
        query = "SELECT * FROM market_data ORDER BY timestamp DESC"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("⚠️ A tabela está vazia. Rode o robô (main.py) primeiro!")
            return

        # 3. Salva no arquivo
        nome_arquivo = "dados_do_robo.csv"
        df.to_csv(nome_arquivo, index=False)
        
        print(f"✅ SUCESSO! Foram exportadas {len(df)} linhas.")
        print(f"📂 O arquivo foi criado na sua pasta: {nome_arquivo}")
        print("👉 Você pode abrir este arquivo clicando nele na pasta do Finder.")
        
    except Exception as e:
        print(f"❌ Erro ao exportar: {e}")

if __name__ == "__main__":
    exportar_para_csv()