import asyncio
import os
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

def testar_binance_oficial():
    print("\n--- 🛡️ VERIFICAÇÃO FINAL (LIB OFICIAL) 🛡️ ---")
    
    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET_KEY")

    if not api_key:
        print("❌ ERRO: Chaves não encontradas no .env")
        return

    print("📡 Conectando via python-binance (Testnet)...")
    
    # Configuração explícita para Testnet de Futuros
    try:
        # A lib python-binance tem um parametro 'testnet=True' que facilita tudo
        client = Client(api_key, secret, testnet=True)
        
        # Tenta pegar dados da conta de Futuros
        # O método futures_account_balance() bate direto na URL certa
        balance_info = client.futures_account_balance()
        
        print("🎉 SUCESSO! Conexão ESTABELECIDA com Binance Futures (Testnet)!")
        
        # Procura saldo em USDT
        usdt_balance = 0
        for asset in balance_info:
            if asset['asset'] == 'USDT':
                usdt_balance = float(asset['balance'])
                print(f"💰 Saldo Encontrado: {usdt_balance:,.2f} USDT")
                break
                
        if usdt_balance == 0:
            print("⚠️  Saldo é zero (normal se acabou de criar), mas a conexão funcionou!")
            
    except BinanceAPIException as e:
        print(f"❌ Erro da API: {e}")
        if "API-key" in str(e):
            print("DICA: O erro 2015/2008 geralmente é resolvido recriando a chave.")
    except Exception as e:
        print(f"❌ Erro Genérico: {e}")

if __name__ == "__main__":
    testar_binance_oficial()