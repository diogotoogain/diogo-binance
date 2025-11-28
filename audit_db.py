import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv

# Carrega configurações
load_dotenv()

async def auditar_banco():
    print("\n--- 🕵️‍♂️ AUDITORIA DO COFRE (PostgreSQL) ---")
    
    # Pega a URL do banco e ajusta para o driver async
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    
    try:
        engine = create_async_engine(db_url)
        
        async with engine.connect() as conn:
            # 1. Quantas linhas temos?
            result_count = await conn.execute(text("SELECT COUNT(*) FROM market_data"))
            total_linhas = result_count.scalar()
            
            print(f"📊 Total de registros salvos: {total_linhas}")
            
            if total_linhas > 0:
                print("\n📉 Últimos 5 registros gravados:")
                # 2. Mostra os últimos 5
                query = text("SELECT timestamp, symbol, close, volume FROM market_data ORDER BY timestamp DESC LIMIT 5")
                result_rows = await conn.execute(query)
                
                print(f"{'HORA (UTC)':<25} | {'PAR':<10} | {'PREÇO':<15} | {'VOLUME'}")
                print("-" * 70)
                
                for row in result_rows:
                    # row é uma tupla (timestamp, symbol, close, volume)
                    print(f"{str(row[0]):<25} | {row[1]:<10} | {row[2]:<15} | {row[3]}")
            else:
                print("⚠️ O banco está vazio. O robô não teve tempo de salvar nada ou deu erro silencioso.")

        await engine.dispose()

    except Exception as e:
        print(f"❌ Erro ao conectar no banco: {e}")

if __name__ == "__main__":
    asyncio.run(auditar_banco())