import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.execution_service import ExecutionService
from app.models.strategy import Strategy, StrategyType, RiskTolerance, StrategyPerformance


@pytest.fixture
def sample_strategy():
    """Create a sample strategy for testing."""
    return Strategy(
        id="test-strategy-1",
        name="Test RSI Strategy",
        description="Test strategy for RSI",
        type=StrategyType.MEAN_REVERSION,
        trading_pairs=["ETH-USDT"],
        risk_tolerance=RiskTolerance.MEDIUM,
        indicators=[
            {
                "name": "RSI",
                "parameters": {"period": 14, "overbought": 70, "oversold": 30}
            }
        ],
        rules=[
            {
                "name": "Buy when oversold",
                "conditions": [
                    {
                        "indicator": "RSI",
                        "operator": "<",
                        "value": 30
                    }
                ],
                "actions": [
                    {
                        "type": "BUY",
                        "asset": "ETH",
                        "amount_type": "percentage",
                        "amount": 50.0
                    }
                ]
            },
            {
                "name": "Sell when overbought",
                "conditions": [
                    {
                        "indicator": "RSI",
                        "operator": ">",
                        "value": 70
                    }
                ],
                "actions": [
                    {
                        "type": "SELL",
                        "asset": "ETH",
                        "amount_type": "percentage",
                        "amount": 100.0
                    }
                ]
            }
        ],
        parameters={
            "stop_loss": 5.0,
            "take_profit": 10.0
        }
    )


@pytest.fixture
def execution_service():
    """Create an execution service instance for testing."""
    with patch('app.services.execution_service.asyncio.create_task') as mock_create_task:
        # Mock create_task to return a MagicMock
        mock_task = MagicMock()
        mock_create_task.return_value = mock_task
        
        service = ExecutionService()
        yield service


@pytest.mark.asyncio
async def test_deploy_strategy(execution_service, sample_strategy):
    """Test deploying a strategy."""
    # Deploy the strategy
    result = await execution_service.deploy_strategy(sample_strategy)
    
    # Check result
    assert result["status"] == "deployed"
    assert result["strategy_id"] == sample_strategy.id
    
    # Check strategy is stored in active strategies
    assert sample_strategy.id in execution_service.active_strategies
    assert execution_service.active_strategies[sample_strategy.id]["strategy"] == sample_strategy
    assert execution_service.active_strategies[sample_strategy.id]["status"] == "starting"
    
    # Check task is created and stored
    assert sample_strategy.id in execution_service.strategy_tasks


@pytest.mark.asyncio
async def test_deploy_strategy_already_deployed(execution_service, sample_strategy):
    """Test deploying a strategy that's already deployed."""
    # First deploy
    await execution_service.deploy_strategy(sample_strategy)
    
    # Try to deploy again
    result = await execution_service.deploy_strategy(sample_strategy)
    
    # Check result
    assert result["status"] == "already_deployed"
    assert result["strategy_id"] == sample_strategy.id


@pytest.mark.asyncio
async def test_stop_strategy(execution_service, sample_strategy):
    """Test stopping a deployed strategy."""
    # First deploy
    await execution_service.deploy_strategy(sample_strategy)
    
    # Mock the task
    task = execution_service.strategy_tasks[sample_strategy.id]
    task.done.return_value = False
    
    # Stop the strategy
    result = await execution_service.stop_strategy(sample_strategy.id)
    
    # Check result
    assert result["status"] == "stopped"
    assert result["strategy_id"] == sample_strategy.id
    
    # Check task is cancelled
    assert task.cancel.called
    
    # Check strategy status is updated
    assert execution_service.active_strategies[sample_strategy.id]["status"] == "stopped"
    
    # Check task is removed from strategy_tasks
    assert sample_strategy.id not in execution_service.strategy_tasks


@pytest.mark.asyncio
async def test_stop_strategy_not_deployed(execution_service):
    """Test stopping a strategy that's not deployed."""
    # Try to stop a non-existent strategy
    result = await execution_service.stop_strategy("non-existent-strategy")
    
    # Check result
    assert result["status"] == "not_deployed"
    assert result["strategy_id"] == "non-existent-strategy"


@pytest.mark.asyncio
async def test_get_strategy_performance(execution_service, sample_strategy):
    """Test getting strategy performance."""
    # First deploy
    await execution_service.deploy_strategy(sample_strategy)
    
    # Set up some test data
    strategy_data = execution_service.active_strategies[sample_strategy.id]
    strategy_data["start_time"] = time.time() - 3600  # 1 hour ago
    strategy_data["positions"] = {"ETH-USDT": {"entry_price": 1000, "amount": 1.0, "timestamp": time.time()}}
    strategy_data["trades"] = [
        {"type": "buy", "trading_pair": "ETH-USDT", "price": 1000, "amount": 1.0, "timestamp": time.time() - 1800}
    ]
    strategy_data["performance"] = {
        "total_return": 0.05,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.02,
        "win_rate": 0.6,
        "num_trades": 5
    }
    
    # Get performance
    performance = await execution_service.get_strategy_performance(sample_strategy.id)
    
    # Check result
    assert isinstance(performance, StrategyPerformance)
    assert performance.strategy_id == sample_strategy.id
    assert performance.total_return == 0.05
    assert "ETH-USDT" in performance.current_positions
    assert len(performance.recent_trades) == 1
    assert performance.metrics["sharpe_ratio"] == 1.2


@pytest.mark.asyncio
async def test_get_strategy_performance_not_deployed(execution_service):
    """Test getting performance for a strategy that's not deployed."""
    # Try to get performance for a non-existent strategy
    with pytest.raises(ValueError, match="Strategy non-existent-strategy is not deployed"):
        await execution_service.get_strategy_performance("non-existent-strategy")


@pytest.mark.asyncio
async def test_run_strategy(execution_service, sample_strategy):
    """Test the _run_strategy method."""
    # Mock dependencies
    with patch('app.services.execution_service.okx_client') as mock_okx_client:
        with patch('app.services.execution_service.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Mock get_market_data
            mock_okx_client.get_market_data = AsyncMock()
            mock_okx_client.get_market_data.return_value = {
                "close": 1000,
                "volume": 100,
                "rsi": 25  # Below oversold threshold to trigger buy
            }
            
            # Mock _evaluate_strategy_rules
            execution_service._evaluate_strategy_rules = AsyncMock()
            execution_service._evaluate_strategy_rules.return_value = {"ETH-USDT": 1}  # Buy signal
            
            # Mock _execute_signal
            execution_service._execute_signal = AsyncMock()
            
            # First deploy
            await execution_service.deploy_strategy(sample_strategy)
            
            # Run the strategy (this would normally be called by create_task)
            # We'll run it directly for testing
            task = asyncio.create_task(execution_service._run_strategy(sample_strategy.id))
            
            # Wait a bit for the task to run
            await asyncio.sleep(0.1)
            
            # Cancel the task to stop the loop
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Check that methods were called
            assert mock_okx_client.get_market_data.called
            assert execution_service._evaluate_strategy_rules.called
            assert execution_service._execute_signal.called
            
            # Check strategy status
            assert execution_service.active_strategies[sample_strategy.id]["status"] in ["running", "stopped"]


@pytest.mark.asyncio
async def test_evaluate_strategy_rules(execution_service, sample_strategy):
    """Test evaluating strategy rules."""
    # Market data with RSI below oversold threshold
    market_data = {
        "ETH-USDT": {
            "close": 1000,
            "volume": 100,
            "rsi": 25  # Below oversold threshold (30)
        }
    }
    
    # Evaluate rules
    signals = await execution_service._evaluate_strategy_rules(sample_strategy, market_data)
    
    # Check signals
    assert "ETH-USDT" in signals
    assert signals["ETH-USDT"] == 1  # Buy signal
    
    # Market data with RSI above overbought threshold
    market_data["ETH-USDT"]["rsi"] = 75  # Above overbought threshold (70)
    
    # Evaluate rules
    signals = await execution_service._evaluate_strategy_rules(sample_strategy, market_data)
    
    # Check signals
    assert signals["ETH-USDT"] == -1  # Sell signal
    
    # Market data with RSI in neutral zone
    market_data["ETH-USDT"]["rsi"] = 50  # Between thresholds
    
    # Evaluate rules
    signals = await execution_service._evaluate_strategy_rules(sample_strategy, market_data)
    
    # Check signals
    assert signals["ETH-USDT"] == 0  # No signal


@pytest.mark.asyncio
async def test_execute_signal_buy(execution_service, sample_strategy):
    """Test executing a buy signal."""
    # Mock okx_client
    with patch('app.services.execution_service.okx_client') as mock_okx_client:
        # Mock execute_swap
        mock_okx_client.execute_swap = AsyncMock()
        mock_okx_client.execute_swap.return_value = {"order_id": "test-order-1"}
        
        # First deploy
        await execution_service.deploy_strategy(sample_strategy)
        
        # Market data
        market_data = {
            "close": 1000,
            "volume": 100
        }
        
        # Execute buy signal
        await execution_service._execute_signal(sample_strategy.id, "ETH-USDT", 1, market_data)
        
        # Check that execute_swap was called
        mock_okx_client.execute_swap.assert_called_once()
        
        # Check that trade was recorded
        strategy_data = execution_service.active_strategies[sample_strategy.id]
        assert len(strategy_data["trades"]) == 1
        assert strategy_data["trades"][0]["type"] == "buy"
        assert strategy_data["trades"][0]["trading_pair"] == "ETH-USDT"
        assert strategy_data["trades"][0]["price"] == 1000
        
        # Check that position was updated
        assert "ETH-USDT" in strategy_data["positions"]
        assert strategy_data["positions"]["ETH-USDT"]["entry_price"] == 1000


@pytest.mark.asyncio
async def test_execute_signal_sell(execution_service, sample_strategy):
    """Test executing a sell signal."""
    # Mock okx_client
    with patch('app.services.execution_service.okx_client') as mock_okx_client:
        # Mock execute_swap
        mock_okx_client.execute_swap = AsyncMock()
        mock_okx_client.execute_swap.return_value = {"order_id": "test-order-2"}
        
        # First deploy
        await execution_service.deploy_strategy(sample_strategy)
        
        # Add a position
        strategy_data = execution_service.active_strategies[sample_strategy.id]
        strategy_data["positions"]["ETH-USDT"] = {
            "entry_price": 900,
            "amount": 1.0,
            "timestamp": time.time() - 3600
        }
        
        # Market data (price increased)
        market_data = {
            "close": 1000,
            "volume": 100
        }
        
        # Mock _update_performance_metrics
        execution_service._update_performance_metrics = MagicMock()
        
        # Execute sell signal
        await execution_service._execute_signal(sample_strategy.id, "ETH-USDT", -1, market_data)
        
        # Check that execute_swap was called
        mock_okx_client.execute_swap.assert_called_once()
        
        # Check that trade was recorded
        assert len(strategy_data["trades"]) == 1
        assert strategy_data["trades"][0]["type"] == "sell"
        assert strategy_data["trades"][0]["trading_pair"] == "ETH-USDT"
        assert strategy_data["trades"][0]["price"] == 1000
        assert strategy_data["trades"][0]["profit"] == 100  # (1000 - 900) * 1.0
        
        # Check that position was removed
        assert "ETH-USDT" not in strategy_data["positions"]
        
        # Check that _update_performance_metrics was called
        execution_service._update_performance_metrics.assert_called_once()


def test_update_performance_metrics(execution_service, sample_strategy):
    """Test updating performance metrics."""
    # First deploy
    asyncio.run(execution_service.deploy_strategy(sample_strategy))
    
    # Add some trades
    strategy_data = execution_service.active_strategies[sample_strategy.id]
    strategy_data["trades"] = [
        {"type": "buy", "trading_pair": "ETH-USDT", "price": 900, "amount": 1.0, "timestamp": time.time() - 7200},
        {"type": "sell", "trading_pair": "ETH-USDT", "price": 1000, "amount": 1.0, "timestamp": time.time() - 3600, "profit": 100},
        {"type": "buy", "trading_pair": "ETH-USDT", "price": 950, "amount": 1.0, "timestamp": time.time() - 1800}
    ]
    
    # Initial performance metrics
    strategy_data["performance"] = {
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "num_trades": 0
    }
    
    # New sell trade with profit
    trade = {
        "type": "sell",
        "trading_pair": "ETH-USDT",
        "price": 1050,
        "amount": 1.0,
        "timestamp": time.time(),
        "profit": 100  # (1050 - 950) * 1.0
    }
    
    # Update metrics
    execution_service._update_performance_metrics(sample_strategy.id, trade)
    
    # Check metrics updated
    assert strategy_data["performance"]["total_return"] == 100
    assert strategy_data["performance"]["num_trades"] == 1
    assert strategy_data["performance"]["win_rate"] == 1.0  # 100% win rate (2/2 profitable sells)
