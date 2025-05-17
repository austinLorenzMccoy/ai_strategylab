import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.rl_service import RLService, TradingEnvironment
from app.models.strategy import Strategy, StrategyType, RiskTolerance, BacktestResult


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
def sample_historical_data():
    """Create sample historical data for testing."""
    # Create 100 days of sample data
    data = []
    for i in range(100):
        # Generate some random price data
        close_price = 1000 + np.random.normal(0, 20)
        volume = 100 + np.random.normal(0, 10)
        
        # Generate RSI values that oscillate between oversold and overbought
        rsi = 50 + 30 * np.sin(i / 10)
        
        data.append({
            "timestamp": f"2025-01-{i+1:02d}T00:00:00Z",
            "open": close_price - 5,
            "high": close_price + 10,
            "low": close_price - 10,
            "close": close_price,
            "volume": volume,
            "rsi": rsi
        })
    
    return data


@pytest.fixture
def trading_env(sample_strategy, sample_historical_data):
    """Create a trading environment for testing."""
    return TradingEnvironment(sample_strategy, sample_historical_data)


@pytest.fixture
def rl_service():
    """Create an RL service instance for testing."""
    return RLService()


def test_trading_env_init(trading_env, sample_strategy, sample_historical_data):
    """Test trading environment initialization."""
    assert trading_env.strategy == sample_strategy
    assert trading_env.historical_data == sample_historical_data
    assert trading_env.current_step == 0
    assert trading_env.portfolio_value == 10000.0
    assert trading_env.position == 0.0
    assert trading_env.cash == 10000.0


def test_trading_env_reset(trading_env):
    """Test trading environment reset."""
    # Change some state
    trading_env.current_step = 10
    trading_env.portfolio_value = 12000.0
    trading_env.position = 1.0
    trading_env.cash = 11000.0
    
    # Reset
    obs, _ = trading_env.reset()
    
    # Check state is reset
    assert trading_env.current_step == 0
    assert trading_env.portfolio_value == 10000.0
    assert trading_env.position == 0.0
    assert trading_env.cash == 10000.0
    
    # Check observation
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)


def test_trading_env_step(trading_env):
    """Test trading environment step."""
    # Take a step
    action = np.array([14, 30, 70])  # RSI parameters
    obs, reward, done, truncated, info = trading_env.step(action)
    
    # Check state updated
    assert trading_env.current_step == 1
    
    # Check observation and info
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "portfolio_value" in info
    assert "position" in info
    assert "cash" in info


def test_trading_env_update_strategy_parameters(trading_env):
    """Test updating strategy parameters."""
    # Initial parameters
    initial_period = trading_env.strategy.indicators[0].parameters["period"]
    initial_oversold = trading_env.strategy.indicators[0].parameters["oversold"]
    initial_overbought = trading_env.strategy.indicators[0].parameters["overbought"]
    
    # Update parameters
    action = np.array([10, 25, 75])
    trading_env._update_strategy_parameters(action)
    
    # Check parameters updated
    assert trading_env.strategy.indicators[0].parameters["period"] == 10
    assert trading_env.strategy.indicators[0].parameters["oversold"] == 25
    assert trading_env.strategy.indicators[0].parameters["overbought"] == 75


def test_trading_env_execute_strategy(trading_env, sample_historical_data):
    """Test executing strategy."""
    # Test with RSI below oversold threshold
    data = sample_historical_data[0].copy()
    data["rsi"] = 25  # Below oversold (30)
    signal = trading_env._execute_strategy(data)
    assert signal == 1  # Buy signal
    
    # Test with RSI above overbought threshold
    data["rsi"] = 75  # Above overbought (70)
    signal = trading_env._execute_strategy(data)
    assert signal == -1  # Sell signal
    
    # Test with RSI in neutral zone
    data["rsi"] = 50  # Between oversold and overbought
    signal = trading_env._execute_strategy(data)
    assert signal == 0  # No signal


def test_trading_env_apply_trading_signal_buy(trading_env, sample_historical_data):
    """Test applying buy signal."""
    # Initial state
    assert trading_env.position == 0.0
    assert trading_env.cash == 10000.0
    
    # Apply buy signal
    data = sample_historical_data[0].copy()
    data["close"] = 1000.0
    reward = trading_env._apply_trading_signal(1, data)  # Buy signal
    
    # Check position and cash updated
    assert trading_env.position > 0.0
    assert trading_env.cash < 10000.0
    assert len(trading_env.trades) == 1
    assert trading_env.trades[0]["type"] == "buy"


def test_trading_env_apply_trading_signal_sell(trading_env, sample_historical_data):
    """Test applying sell signal."""
    # First buy to create a position
    data = sample_historical_data[0].copy()
    data["close"] = 1000.0
    trading_env._apply_trading_signal(1, data)  # Buy signal
    
    # Then sell
    data["close"] = 1100.0  # Price increased
    reward = trading_env._apply_trading_signal(-1, data)  # Sell signal
    
    # Check position and cash updated
    assert trading_env.position == 0.0
    assert trading_env.cash > 10000.0  # Made profit
    assert len(trading_env.trades) == 2
    assert trading_env.trades[1]["type"] == "sell"


def test_trading_env_calculate_performance_metrics(trading_env, sample_historical_data):
    """Test calculating performance metrics."""
    # Create some trades
    data = sample_historical_data[0].copy()
    data["close"] = 1000.0
    trading_env._apply_trading_signal(1, data)  # Buy
    
    data["close"] = 1100.0
    trading_env._apply_trading_signal(-1, data)  # Sell with profit
    
    data["close"] = 1000.0
    trading_env._apply_trading_signal(1, data)  # Buy again
    
    data["close"] = 950.0
    trading_env._apply_trading_signal(-1, data)  # Sell with loss
    
    # Calculate metrics
    metrics = trading_env.calculate_performance_metrics()
    
    # Check metrics
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
    assert "num_trades" in metrics
    assert metrics["num_trades"] == 2  # Two sell trades


@pytest.mark.asyncio
async def test_rl_service_backtest_strategy(rl_service, sample_strategy, sample_historical_data):
    """Test backtesting a strategy."""
    with patch('app.services.rl_service.TradingEnvironment') as mock_env:
        # Mock the environment
        env_instance = MagicMock()
        mock_env.return_value = env_instance
        
        # Mock reset and step methods
        env_instance.reset.return_value = (np.zeros(10), {})
        env_instance.step.return_value = (np.zeros(10), 0.0, True, False, {})
        
        # Mock performance metrics
        env_instance.calculate_performance_metrics.return_value = {
            "total_return": 0.1,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.05,
            "win_rate": 0.6,
            "num_trades": 10
        }
        
        # Mock trades
        env_instance.trades = []
        
        # Call backtest_strategy
        result = await rl_service.backtest_strategy(sample_strategy, sample_historical_data)
        
        # Check TradingEnvironment was created with the right arguments
        mock_env.assert_called_with(sample_strategy, sample_historical_data)
        
        # Check result
        assert isinstance(result, BacktestResult)
        assert result.strategy_id == sample_strategy.id
        assert result.total_return == 0.1
        assert result.sharpe_ratio == 1.5
        assert result.max_drawdown == 0.05
        assert result.win_rate == 0.6
        assert result.num_trades == 10
