import pytest
import sys
sys.path.insert(0, '.')

from unittest.mock import AsyncMock, MagicMock, patch
from src.execution.trade_executor import TradeExecutor


class TestTradeExecutor:
    def setup_method(self):
        self.mock_client = MagicMock()
        self.mock_position_manager = MagicMock()
        self.mock_risk_manager = MagicMock()
        self.mock_notifier = MagicMock()
        
        self.executor = TradeExecutor(
            client=self.mock_client,
            position_manager=self.mock_position_manager,
            risk_manager=self.mock_risk_manager,
            symbol="BTCUSDT",
            min_confidence="HIGH",
            execution_enabled=True,
            notifier=self.mock_notifier
        )

    @pytest.mark.asyncio
    async def test_open_position_zero_avg_price_uses_fallback(self):
        """Test that zero avgPrice from Binance uses the price parameter as fallback."""
        self.mock_client.futures_create_order = AsyncMock(return_value={
            'avgPrice': '0',  # Binance sometimes returns "0"
            'orderId': 12345
        })
        self.mock_notifier.notify_trade_open = AsyncMock()
        
        result = await self.executor._open_position('BUY', 0.1, 50000.0, 49500.0, 51000.0)
        
        assert result is True
        # Verify position was opened with the fallback price (50000)
        self.mock_position_manager.open_position.assert_called_once()
        call_args = self.mock_position_manager.open_position.call_args
        assert call_args[0][2] == 50000.0  # fill price should be fallback

    @pytest.mark.asyncio
    async def test_open_position_zero_avg_price_numeric_uses_fallback(self):
        """Test that numeric zero avgPrice from Binance uses the price parameter as fallback."""
        self.mock_client.futures_create_order = AsyncMock(return_value={
            'avgPrice': 0,  # Binance sometimes returns 0 as number
            'orderId': 12345
        })
        self.mock_notifier.notify_trade_open = AsyncMock()
        
        result = await self.executor._open_position('BUY', 0.1, 50000.0, 49500.0, 51000.0)
        
        assert result is True
        self.mock_position_manager.open_position.assert_called_once()
        call_args = self.mock_position_manager.open_position.call_args
        assert call_args[0][2] == 50000.0

    @pytest.mark.asyncio
    async def test_open_position_missing_avg_price_uses_fallback(self):
        """Test that missing avgPrice from Binance uses the price parameter as fallback."""
        self.mock_client.futures_create_order = AsyncMock(return_value={
            'orderId': 12345
            # No avgPrice key
        })
        self.mock_notifier.notify_trade_open = AsyncMock()
        
        result = await self.executor._open_position('BUY', 0.1, 50000.0, 49500.0, 51000.0)
        
        assert result is True
        self.mock_position_manager.open_position.assert_called_once()
        call_args = self.mock_position_manager.open_position.call_args
        assert call_args[0][2] == 50000.0

    @pytest.mark.asyncio
    async def test_open_position_both_prices_zero_returns_false(self):
        """Test that when both avgPrice and price are zero/invalid, position is not opened."""
        self.mock_client.futures_create_order = AsyncMock(return_value={
            'avgPrice': '0',
            'orderId': 12345
        })
        
        # Both avgPrice and price parameter are zero
        result = await self.executor._open_position('BUY', 0.1, 0.0, 49500.0, 51000.0)
        
        assert result is False
        self.mock_position_manager.open_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_open_position_valid_avg_price(self):
        """Test that valid avgPrice from Binance is used correctly."""
        self.mock_client.futures_create_order = AsyncMock(return_value={
            'avgPrice': '50500.00',
            'orderId': 12345
        })
        self.mock_notifier.notify_trade_open = AsyncMock()
        
        result = await self.executor._open_position('BUY', 0.1, 50000.0, 49500.0, 51000.0)
        
        assert result is True
        self.mock_position_manager.open_position.assert_called_once()
        call_args = self.mock_position_manager.open_position.call_args
        assert call_args[0][2] == 50500.0  # Should use actual avgPrice
