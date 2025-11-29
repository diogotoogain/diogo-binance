"""Tests for BinanceConnector demo mode functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import importlib.util
sys.path.insert(0, '.')


def load_binance_connector():
    """Helper function to load BinanceConnector module with mocked dependencies."""
    spec = importlib.util.spec_from_file_location(
        "binance_connector", 
        "src/data/binance_connector.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['src.core.event_bus'] = MagicMock()
    spec.loader.exec_module(module)
    return module


class TestBinanceConnector:
    def setup_method(self):
        # Create a mock event bus
        self.mock_event_bus = MagicMock()
        self.mock_event_bus.publish = AsyncMock()

    def test_initialization_default(self):
        """Test default initialization (non-demo mode)"""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
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
            module = load_binance_connector()
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
            module = load_binance_connector()
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
            module = load_binance_connector()
            BinanceConnector = module.BinanceConnector
            
            # Mock AsyncClient.create after module is loaded
            mock_async_client = MagicMock()
            mock_async_client.create = AsyncMock(return_value=mock_client)
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
            module = load_binance_connector()
            BinanceConnector = module.BinanceConnector
            
            # Mock AsyncClient.create after module is loaded
            mock_async_client = MagicMock()
            mock_async_client.create = AsyncMock(return_value=mock_client)
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


class TestWebSocketHealthCheck:
    """Tests for WebSocketHealthCheck class."""
    
    def test_initialization(self):
        """Test WebSocketHealthCheck initialization."""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            WebSocketHealthCheck = module.WebSocketHealthCheck
            
            hc = WebSocketHealthCheck(timeout_seconds=30)
            assert hc.timeout_seconds == 30
            assert hc.last_message_time is None
            assert hc.last_valid_price is None

    def test_on_message_updates_timestamp(self):
        """Test that on_message updates the last_message_time."""
        import time
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            WebSocketHealthCheck = module.WebSocketHealthCheck
            
            hc = WebSocketHealthCheck(timeout_seconds=30)
            assert hc.last_message_time is None
            
            hc.on_message()
            assert hc.last_message_time is not None
            assert time.time() - hc.last_message_time < 1  # Should be very recent

    def test_on_message_updates_price(self):
        """Test that on_message updates the last_valid_price with valid prices."""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            WebSocketHealthCheck = module.WebSocketHealthCheck
            
            hc = WebSocketHealthCheck(timeout_seconds=30)
            
            # Valid price
            hc.on_message(price=50000)
            assert hc.last_valid_price == 50000
            
            # Invalid prices should not update
            hc.on_message(price=0)
            assert hc.last_valid_price == 50000
            
            hc.on_message(price=-100)
            assert hc.last_valid_price == 50000
            
            hc.on_message(price=None)
            assert hc.last_valid_price == 50000

    def test_is_healthy_returns_false_initially(self):
        """Test that is_healthy returns False when no messages received."""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            WebSocketHealthCheck = module.WebSocketHealthCheck
            
            hc = WebSocketHealthCheck(timeout_seconds=30)
            assert hc.is_healthy() is False

    def test_is_healthy_returns_true_after_message(self):
        """Test that is_healthy returns True after receiving a message."""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            WebSocketHealthCheck = module.WebSocketHealthCheck
            
            hc = WebSocketHealthCheck(timeout_seconds=30)
            hc.on_message()
            assert hc.is_healthy() is True

    def test_is_healthy_returns_false_after_timeout(self):
        """Test that is_healthy returns False after timeout."""
        import time
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            WebSocketHealthCheck = module.WebSocketHealthCheck
            
            hc = WebSocketHealthCheck(timeout_seconds=1)  # 1 second timeout
            hc.on_message()
            
            # Simulate timeout by setting last_message_time in the past
            hc.last_message_time = time.time() - 2  # 2 seconds ago
            assert hc.is_healthy() is False

    def test_connector_has_health_check(self):
        """Test that BinanceConnector has health_check attribute."""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            BinanceConnector = module.BinanceConnector
            
            mock_event_bus = MagicMock()
            connector = BinanceConnector(
                api_key="test_key",
                api_secret="test_secret",
                event_bus=mock_event_bus
            )
            
            assert hasattr(connector, 'health_check')
            assert connector.health_check is not None

    def test_connector_is_websocket_healthy(self):
        """Test that BinanceConnector.is_websocket_healthy() works."""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            BinanceConnector = module.BinanceConnector
            
            mock_event_bus = MagicMock()
            connector = BinanceConnector(
                api_key="test_key",
                api_secret="test_secret",
                event_bus=mock_event_bus
            )
            
            # Initially unhealthy
            assert connector.is_websocket_healthy() is False
            
            # After receiving a message
            connector.health_check.on_message(price=50000)
            assert connector.is_websocket_healthy() is True

    def test_connector_get_last_valid_price(self):
        """Test that BinanceConnector.get_last_valid_price() works."""
        with patch.dict('sys.modules', {'src.core': MagicMock(), 'src.core.event_bus': MagicMock()}):
            module = load_binance_connector()
            BinanceConnector = module.BinanceConnector
            
            mock_event_bus = MagicMock()
            connector = BinanceConnector(
                api_key="test_key",
                api_secret="test_secret",
                event_bus=mock_event_bus
            )
            
            # Initially None
            assert connector.get_last_valid_price() is None
            
            # After receiving a message with price
            connector.health_check.on_message(price=50000)
            assert connector.get_last_valid_price() == 50000
