#!/usr/bin/env python3
"""
Validate Config Script

CLI para validar arquivo de configuração e exibir parâmetros.

Uso:
    python validate_config.py
    python validate_config.py --config path/to/config.yaml
    python validate_config.py --count-params
"""

import argparse
import sys
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import ConfigLoader, KillSwitchDisabledError, ConfigValidationError
from src.config.schema import get_all_optimizable_params, count_optimizable_params


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Validar configuração do V2 Trading Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho para arquivo de configuração YAML"
    )
    
    parser.add_argument(
        "--count-params",
        action="store_true",
        help="Contar parâmetros otimizáveis"
    )
    
    parser.add_argument(
        "--show-params",
        action="store_true",
        help="Mostrar todos os parâmetros otimizáveis"
    )
    
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        help="Mostrar apenas seção específica"
    )
    
    return parser.parse_args()


def print_header(text: str) -> None:
    """Imprime header formatado."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def validate_config(config_path: str = None) -> bool:
    """
    Valida configuração.
    
    Args:
        config_path: Caminho opcional para config
        
    Returns:
        True se válido, False caso contrário
    """
    print_header("🔍 VALIDAÇÃO DE CONFIGURAÇÃO")
    
    try:
        # Reset singleton para forçar recarga
        ConfigLoader.reset()
        
        # Carrega config
        config = ConfigLoader(config_path)
        
        print("\n✅ Configuração carregada com sucesso!")
        print(f"   📁 Arquivo: {config._config_path}")
        
        # Mostra parâmetros principais
        print("\n📊 PARÂMETROS PRINCIPAIS:")
        print(f"   Modo: {config.get('environment.mode')}")
        print(f"   Símbolo: {config.get('market.symbol')}")
        print(f"   Demo Header: {config.get('environment.use_demo_header')}")
        
        print("\n⚙️ RISCO:")
        print(f"   Risk per Trade: {config.get('risk.risk_per_trade_pct')}%")
        print(f"   Max Leverage: {config.get('risk.max_leverage')}x")
        print(f"   Max Daily Loss: {config.get('risk.max_daily_loss_pct')}%")
        print(f"   Max Drawdown: {config.get('risk.max_drawdown_pct')}%")
        
        print("\n🛡️ KILL SWITCH:")
        kill_switch = config.get_section('risk').get('kill_switch', {})
        print(f"   Enabled: {kill_switch.get('enabled', False)} ✅")
        print(f"   Max Loss Trigger: {kill_switch.get('max_loss_trigger_pct')}%")
        print(f"   Pause Duration: {kill_switch.get('pause_duration_hours')}h")
        
        print("\n🌐 WEBSOCKET:")
        print(f"   Buffer Size: {config.get('websocket.buffer_size')} mensagens")
        
        print("\n📈 ESTRATÉGIAS HABILITADAS:")
        strategies = config.get_section('strategies')
        for name, cfg in strategies.items():
            status = "✅" if cfg.get('enabled', False) else "❌"
            print(f"   {status} {name}")
        
        return True
        
    except KillSwitchDisabledError as e:
        print(f"\n🚨 ERRO CRÍTICO: {e}")
        print("\n⚠️  O kill switch DEVE estar SEMPRE ativo!")
        print("   Corrija em: risk.kill_switch.enabled: true")
        return False
        
    except ConfigValidationError as e:
        print(f"\n❌ Erro de validação: {e}")
        return False
        
    except FileNotFoundError as e:
        print(f"\n❌ Arquivo não encontrado: {e}")
        return False
        
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        return False


def count_params() -> None:
    """Conta e exibe número de parâmetros otimizáveis."""
    print_header("📊 CONTAGEM DE PARÂMETROS OTIMIZÁVEIS")
    
    params = get_all_optimizable_params()
    total = len(params)
    
    # Conta por seção
    sections = {}
    for p in params:
        section = p.name.split('.')[0]
        sections[section] = sections.get(section, 0) + 1
    
    print(f"\n📈 TOTAL: {total} parâmetros\n")
    print("Por seção:")
    
    for section, count in sorted(sections.items()):
        bar = "█" * (count // 2)
        print(f"   {section:25} {count:3} {bar}")
    
    print(f"\n   {'TOTAL':25} {total:3}")


def show_params(section: str = None) -> None:
    """Mostra parâmetros otimizáveis."""
    params = get_all_optimizable_params()
    
    if section:
        params = [p for p in params if p.name.startswith(section)]
        print_header(f"📋 PARÂMETROS: {section}")
    else:
        print_header("📋 TODOS OS PARÂMETROS OTIMIZÁVEIS")
    
    print(f"\nTotal: {len(params)} parâmetros\n")
    
    current_section = None
    for p in params:
        # Header de seção
        parts = p.name.split('.')
        sec = '.'.join(parts[:2]) if len(parts) > 2 else parts[0]
        
        if sec != current_section:
            current_section = sec
            print(f"\n[{sec}]")
        
        # Parâmetro
        type_str = p.param_type.value
        
        if p.param_type.value in ('float', 'int'):
            range_str = f"[{p.low}, {p.high}]"
        elif p.choices:
            range_str = str(p.choices)
        else:
            range_str = ""
        
        print(f"  {p.name:50} {type_str:12} {range_str}")


def main():
    """Função principal."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("  🤖 V2 TRADING BOT - VALIDAÇÃO DE CONFIGURAÇÃO")
    print("="*60)
    
    # Valida config
    valid = validate_config(args.config)
    
    # Conta parâmetros
    if args.count_params:
        count_params()
    
    # Mostra parâmetros
    if args.show_params:
        show_params(args.section)
    
    # Status final
    print("\n" + "="*60)
    if valid:
        print("  ✅ CONFIGURAÇÃO VÁLIDA")
    else:
        print("  ❌ CONFIGURAÇÃO INVÁLIDA")
    print("="*60 + "\n")
    
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
