"""Tests for BinanceConnector demo mode functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
sys.path.insert(0, '.')


class TestBinanceConnector:
    def setup_method(self):
        # Create a mock event bus
        self.mock_event_bus = MagicMock()
        self.mock_event_bus.publish = AsyncMock()

    def test_initialization_default(self):
        """Test default initialization (non-demo mode)"""
        # Patch the imports to avoid loading the broken meta_controller
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            # Import directly without going through __init__.py
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "binance_connector", 
                "src/data/binance_connector.py"
            )
            module = importlib.util.module_from_spec(spec)
            
            # Mock the EventBus in the module
            sys.modules['src.core.event_bus'] = MagicMock()
            spec.loader.exec_module(module)
            
            BinanceConnector = module.BinanceConnector
            
            connector = BinanceConnector(
                api_key="test_key",
                api_secret="test_secret",
                event_bus=self.mock_event_bus
            )
            assert connector.api_key == "test_key"
            assert connector.api_secret == "test_secret"
            assert connector.demo_mode is False
            assert connector.client is None
            assert connector.bsm is None

    def test_initialization_demo_mode(self):
        """Test initialization with demo_mode=True"""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "binance_connector", 
                "src/data/binance_connector.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules['src.core.event_bus'] = MagicMock()
            spec.loader.exec_module(module)
            
            BinanceConnector = module.BinanceConnector
            
            connector = BinanceConnector(
                api_key="test_key",
                api_secret="test_secret",
                event_bus=self.mock_event_bus,
                demo_mode=True
            )
            assert connector.demo_mode is True

    def test_initialization_production_mode(self):
        """Test initialization with demo_mode=False"""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "binance_connector", 
                "src/data/binance_connector.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules['src.core.event_bus'] = MagicMock()
            spec.loader.exec_module(module)
            
            BinanceConnector = module.BinanceConnector
            
            connector = BinanceConnector(
                api_key="test_key",
                api_secret="test_secret",
                event_bus=self.mock_event_bus,
                demo_mode=False
            )
            assert connector.demo_mode is False

    @pytest.mark.asyncio
    async def test_connect_demo_mode_sets_header(self):
        """Test that demo mode sets the X-MBX-DEMO header"""
        # Create a mock client with session
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_client.session = mock_session
        
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "binance_connector", 
                "src/data/binance_connector.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules['src.core.event_bus'] = MagicMock()
            
            # Mock AsyncClient before loading module
            mock_async_client = MagicMock()
            mock_async_client.create = AsyncMock(return_value=mock_client)
            module.AsyncClient = mock_async_client
            
            spec.loader.exec_module(module)
            
            BinanceConnector = module.BinanceConnector
            
            # Manually set the mock AsyncClient on the module
            module.AsyncClient = mock_async_client
            
            connector = BinanceConnector(
                api_key="test_key",
                api_secret="test_secret",
                event_bus=self.mock_event_bus,
                demo_mode=True
            )
            
            await connector.connect()
            
            # Verify that AsyncClient.create was called with testnet=False
            mock_async_client.create.assert_called_once_with(
                "test_key", "test_secret", testnet=False
            )
            
            # Verify that the X-MBX-DEMO header was set
            assert 'X-MBX-DEMO' in mock_session.headers
            assert mock_session.headers['X-MBX-DEMO'] == 'true'

    @pytest.mark.asyncio
    async def test_connect_production_mode_no_demo_header(self):
        """Test that production mode does not set the X-MBX-DEMO header"""
        # Create a mock client with session
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_client.session = mock_session
        
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "binance_connector", 
                "src/data/binance_connector.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules['src.core.event_bus'] = MagicMock()
            
            # Mock AsyncClient before loading module
            mock_async_client = MagicMock()
            mock_async_client.create = AsyncMock(return_value=mock_client)
            
            spec.loader.exec_module(module)
            
            BinanceConnector = module.BinanceConnector
            
            # Manually set the mock AsyncClient on the module
            module.AsyncClient = mock_async_client
            
            connector = BinanceConnector(
                api_key="test_key",
                api_secret="test_secret",
                event_bus=self.mock_event_bus,
                demo_mode=False
            )
            
            await connector.connect()
            
            # Verify that AsyncClient.create was called with testnet=False
            mock_async_client.create.assert_called_once_with(
                "test_key", "test_secret", testnet=False
            )
            
            # Verify that the X-MBX-DEMO header was NOT set
            assert 'X-MBX-DEMO' not in mock_session.headers
