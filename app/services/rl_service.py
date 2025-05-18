import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

# RL libraries
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Project imports
from app.models.strategy import Strategy, BacktestResult, TradeSignal
from app.core.config import get_settings
from app.core.okx_client import okx_client
from app.services.websocket_service import MessagePublisher
from app.utils.data_utils import calculate_indicators, normalize_data

logger = logging.getLogger(__name__)
settings = get_settings()


class TradingEnvironment(gym.Env):
    """Advanced trading environment for RL optimization with realistic market simulation."""
    
    def __init__(self, 
                 strategy: Strategy, 
                 historical_data: List[Dict[str, Any]],
                 initial_balance: float = 10000.0,
                 trading_fee: float = 0.001,  # 0.1% trading fee
                 slippage: float = 0.001,     # 0.1% slippage
                 window_size: int = 30,       # Number of past candles to include in state
                 reward_function: str = 'sharpe',  # Options: 'returns', 'sharpe', 'sortino', 'calmar'
                 risk_adjusted: bool = True,   # Apply risk management
                 max_position_size: float = 0.2,  # Maximum position size as fraction of portfolio
                 websocket_channel: Optional[str] = None  # Channel to publish training progress
                ):
        """Initialize the advanced trading environment.
        
        Args:
            strategy: The trading strategy to optimize
            historical_data: Historical market data for backtesting
            initial_balance: Initial portfolio balance
            trading_fee: Trading fee as a fraction (e.g., 0.001 for 0.1%)
            slippage: Slippage as a fraction (e.g., 0.001 for 0.1%)
            window_size: Number of past candles to include in state
            reward_function: Type of reward function to use
            risk_adjusted: Whether to apply risk management
            max_position_size: Maximum position size as fraction of portfolio
            websocket_channel: Channel to publish training progress
        """
        super().__init__()
        
        # Core parameters
        self.strategy = strategy
        self.historical_data = self._preprocess_data(historical_data)
        self.window_size = window_size
        self.current_step = 0
        self.total_steps = len(historical_data)
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.reward_function = reward_function
        self.risk_adjusted = risk_adjusted
        self.max_position_size = max_position_size
        self.websocket_channel = websocket_channel
        
        # Define action space based on strategy parameters to optimize
        # This is a more flexible approach that adapts to the strategy
        self.action_parameters = self._get_strategy_parameters()
        action_low = np.array([param['min'] for param in self.action_parameters])
        action_high = np.array([param['max'] for param in self.action_parameters])
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # Define observation space
        # Include price data, technical indicators, and portfolio state
        feature_count = self._calculate_feature_count()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_count,),
            dtype=np.float32
        )
        
        # Trading state
        self.portfolio_value = self.initial_balance
        self.position = 0.0  # Current position size in base currency
        self.cash = self.initial_balance  # Available cash
        self.entry_price = 0.0  # Price at which position was entered
        
        # Performance tracking
        self.returns = []
        self.daily_returns = []
        self.trades = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance
        
        # Risk metrics
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        
        # Training progress
        self.episode_count = 0
        self.best_reward = -np.inf
        self.best_params = None
    
    def _preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of dictionaries to DataFrame and calculate indicators."""
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp':
                    df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1h')
                else:
                    df[col] = np.random.random(len(df)) * 100  # Placeholder data
        
        # Calculate technical indicators based on strategy
        df = calculate_indicators(df, self.strategy)
        
        # Normalize data to improve RL training
        df = normalize_data(df)
        
        return df
    
    def _get_strategy_parameters(self) -> List[Dict[str, Any]]:
        """Extract strategy parameters that can be optimized by RL."""
        parameters = []
        
        # Extract parameters from indicators
        for indicator in self.strategy.indicators:
            for param_name, param_value in indicator.parameters.items():
                # Define parameter bounds based on type
                if param_name == 'period':
                    parameters.append({
                        'name': f"{indicator.name}_{param_name}",
                        'min': 2,
                        'max': 50,
                        'indicator': indicator.name,
                        'param': param_name
                    })
                elif param_name in ['oversold', 'overbought']:
                    parameters.append({
                        'name': f"{indicator.name}_{param_name}",
                        'min': 10 if param_name == 'oversold' else 70,
                        'max': 40 if param_name == 'oversold' else 95,
                        'indicator': indicator.name,
                        'param': param_name
                    })
                elif param_name in ['fast_period', 'slow_period']:
                    min_val = 3 if param_name == 'fast_period' else 10
                    max_val = 15 if param_name == 'fast_period' else 50
                    parameters.append({
                        'name': f"{indicator.name}_{param_name}",
                        'min': min_val,
                        'max': max_val,
                        'indicator': indicator.name,
                        'param': param_name
                    })
        
        # If no parameters found, add default ones
        if not parameters:
            parameters = [
                {'name': 'default_threshold', 'min': 0.1, 'max': 0.9, 'indicator': 'default', 'param': 'threshold'},
                {'name': 'position_size', 'min': 0.1, 'max': self.max_position_size, 'indicator': 'risk', 'param': 'position_size'}
            ]
        
        return parameters
    
    def _calculate_feature_count(self) -> int:
        """Calculate the number of features in the observation space."""
        # Count basic price features (OHLCV)
        base_features = 5
        
        # Count technical indicators
        indicator_count = sum(len(indicator.parameters) + 1 for indicator in self.strategy.indicators)
        
        # Count portfolio state features (portfolio value, position, cash, etc.)
        portfolio_features = 5
        
        # Total features
        return base_features + indicator_count + portfolio_features
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset portfolio state
        self.portfolio_value = self.initial_balance
        self.position = 0.0
        self.cash = self.initial_balance
        self.entry_price = 0.0
        
        # Reset performance tracking
        self.returns = []
        self.daily_returns = []
        self.trades = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance
        
        # Reset risk metrics
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        
        # Increment episode counter
        self.episode_count += 1
        
        # Get initial observation
        observation = self._get_observation()
        
        # Return observation and info
        return observation, {}
    
    def step(self, action):
        """Take a step in the environment with advanced features.
        
        Args:
            action: Strategy parameters to optimize
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Update strategy parameters based on action
        self._update_strategy_parameters(action)
        
        # Get current market data
        current_data = self.historical_data.iloc[self.current_step]
        
        # Execute strategy with updated parameters
        signal = self._execute_strategy(current_data)
        
        # Apply trading signal with risk management
        reward = self._apply_trading_signal(signal, current_data)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Publish progress to WebSocket if channel is configured
        if self.websocket_channel and self.current_step % 100 == 0:
            self._publish_progress()
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.total_steps - 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Calculate info dict with detailed metrics
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "cash": self.cash,
            "step": self.current_step,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "total_trades": len(self.trades),
            "params": {param['name']: action[i] for i, param in enumerate(self.action_parameters)}
        }
        
        # Save best parameters if this is the best performance
        if done and self.portfolio_value > self.best_reward:
            self.best_reward = self.portfolio_value
            self.best_params = {param['name']: action[i] for i, param in enumerate(self.action_parameters)}
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        """Get current market observation with lookback window."""
        if self.current_step >= self.total_steps:
            return np.zeros(self.observation_space.shape)
        
        # Get window of data
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        window_data = self.historical_data.iloc[start_idx:end_idx]
        
        # Extract features from window data
        price_features = window_data[['open', 'high', 'low', 'close', 'volume']].values.flatten()
        
        # Extract technical indicators
        indicator_features = []
        for indicator in self.strategy.indicators:
            if indicator.name in window_data.columns:
                indicator_features.append(window_data[indicator.name].values[-1])
        
        # Portfolio state features
        portfolio_features = np.array([
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            self.position,                               # Current position size
            self.cash / self.initial_balance,            # Normalized cash
            self.current_drawdown,                       # Current drawdown
            1 if self.position > 0 else 0                # Position flag (long/none)
        ])
        
        # Combine all features
        features = np.concatenate([
            price_features[-5:],  # Last 5 price features
            np.array(indicator_features),
            portfolio_features
        ])
        
        # Ensure correct shape
        if len(features) < self.observation_space.shape[0]:
            features = np.pad(features, (0, self.observation_space.shape[0] - len(features)))
        elif len(features) > self.observation_space.shape[0]:
            features = features[:self.observation_space.shape[0]]
        
        return features.astype(np.float32)
    
    def _get_observation(self):
        """Get current market observation."""
        if self.current_step >= self.total_steps:
            return np.zeros(self.observation_space.shape)
        
        # Extract features from current market data
        data = self.historical_data[self.current_step]
        
        # Example features: price, volume, technical indicators, etc.
        # In a real implementation, calculate these from the data
        features = np.array([
            data.get("close", 0),
            data.get("volume", 0),
            # Add more features as needed
            0, 0, 0, 0, 0, 0, 0, 0  # Placeholder for other features
        ], dtype=np.float32)[:self.observation_space.shape[0]]
        
        return features
    
    def _update_strategy_parameters(self, action):
        """Update strategy parameters based on RL action."""
        # Map action values to strategy parameters
        for i, param_config in enumerate(self.action_parameters):
            # Get parameter value from action
            param_value = float(action[i])
            
            # Round to integer if needed
            if param_config['param'] in ['period', 'fast_period', 'slow_period']:
                param_value = int(param_value)
            
            # Update the parameter in the strategy
            for indicator in self.strategy.indicators:
                if indicator.name == param_config['indicator']:
                    indicator.parameters[param_config['param']] = param_value
            
            # Handle special case for position sizing
            if param_config['param'] == 'position_size':
                self.max_position_size = param_value
    
    def _execute_strategy(self, data):
        """Execute strategy with current parameters and market data."""
        # Default signal: 0 = hold, 1 = buy, -1 = sell
        signal = 0
        
        # Evaluate all indicators and rules in the strategy
        for indicator in self.strategy.indicators:
            if indicator.name == "RSI":
                # Get RSI value from preprocessed data
                rsi_value = data.get("RSI", 50)
                
                if rsi_value < indicator.parameters.get("oversold", 30):
                    signal = 1  # Buy signal
                elif rsi_value > indicator.parameters.get("overbought", 70):
                    signal = -1  # Sell signal
                    
            elif indicator.name == "MACD":
                # Get MACD values from preprocessed data
                macd = data.get("MACD", 0)
                macd_signal = data.get("MACD_signal", 0)
                
                if macd > macd_signal:
                    signal = 1  # Buy signal
                elif macd < macd_signal:
                    signal = -1  # Sell signal
                    
            elif indicator.name == "Bollinger":
                # Get Bollinger Band values
                price = data.get("close", 0)
                upper_band = data.get("bb_upper", price * 1.1)
                lower_band = data.get("bb_lower", price * 0.9)
                
                if price < lower_band:
                    signal = 1  # Buy signal
                elif price > upper_band:
                    signal = -1  # Sell signal
        
        # Apply any custom rules from the strategy
        for rule in self.strategy.rules:
            # Implement rule evaluation logic here
            pass
        
        return signal
    
    def _apply_trading_signal(self, signal, data):
        """Apply trading signal with risk management and calculate reward."""
        price = data.get("close", 0)
        prev_portfolio_value = self.portfolio_value
        
        # Apply slippage to price
        buy_price = price * (1 + self.slippage)
        sell_price = price * (1 - self.slippage)
        
        # Execute trades based on signal with risk management
        if signal == 1 and self.position == 0:  # Buy
            # Calculate position size with risk management
            if self.risk_adjusted:
                # Use volatility-adjusted position sizing
                volatility = data.get("volatility", 0.02)  # Default 2% volatility
                risk_factor = min(1.0, 0.1 / max(0.001, volatility))  # Lower volatility = larger position
                position_size = min(self.max_position_size, self.max_position_size * risk_factor)
            else:
                position_size = self.max_position_size
            
            # Calculate amount to buy
            amount = self.cash * position_size
            shares = amount / buy_price
            
            # Apply trading fee
            fee = amount * self.trading_fee
            amount_with_fee = amount + fee
            
            # Execute trade if we have enough cash
            if amount_with_fee <= self.cash:
                self.position = shares
                self.cash -= amount_with_fee
                self.entry_price = buy_price
                
                # Record trade
                self.trades.append({
                    "type": "buy",
                    "price": buy_price,
                    "amount": amount,
                    "shares": shares,
                    "fee": fee,
                    "step": self.current_step,
                    "timestamp": data.get("timestamp", self.current_step)
                })
            
        elif signal == -1 and self.position > 0:  # Sell
            # Calculate amount from selling position
            shares = self.position
            amount = shares * sell_price
            
            # Apply trading fee
            fee = amount * self.trading_fee
            amount_after_fee = amount - fee
            
            # Execute trade
            self.cash += amount_after_fee
            self.position = 0
            
            # Record trade with P&L
            profit_loss = (sell_price - self.entry_price) * shares - fee
            profit_loss_pct = profit_loss / (self.entry_price * shares) if self.entry_price > 0 else 0
            
            self.trades.append({
                "type": "sell",
                "price": sell_price,
                "amount": amount,
                "shares": shares,
                "fee": fee,
                "step": self.current_step,
                "timestamp": data.get("timestamp", self.current_step),
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct
            })
        
        # Update portfolio value
        self.portfolio_value = self.cash + (self.position * price)
        
        # Update peak value and drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Calculate return
        step_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        self.returns.append(step_return)
        
        # Calculate reward based on selected reward function
        if self.reward_function == 'returns':
            reward = step_return
        elif self.reward_function == 'sharpe':
            reward = self._calculate_sharpe_ratio()
        elif self.reward_function == 'sortino':
            reward = self._calculate_sortino_ratio()
        elif self.reward_function == 'calmar':
            reward = self._calculate_calmar_ratio()
        else:
            reward = step_return  # Default to returns
        
        return reward
    
    def _update_performance_metrics(self):
        """Update performance metrics after each step."""
        # Only update if we have enough data
        if len(self.returns) < 2:
            return
        
        # Calculate volatility (standard deviation of returns)
        self.volatility = np.std(self.returns[-30:]) if len(self.returns) >= 30 else np.std(self.returns)
        
        # Calculate Sharpe ratio
        self.sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Calculate Sortino ratio
        self.sortino_ratio = self._calculate_sortino_ratio()
        
        # Calculate Calmar ratio
        self.calmar_ratio = self._calculate_calmar_ratio()
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0):
        """Calculate Sharpe ratio."""
        if len(self.returns) < 2:
            return 0.0
        
        # Calculate mean return and standard deviation
        mean_return = np.mean(self.returns)
        std_return = np.std(self.returns)
        
        # Avoid division by zero
        if std_return == 0:
            return 0.0
        
        # Calculate annualized Sharpe ratio (assuming daily returns)
        # Multiply by sqrt(252) for annualization (252 trading days in a year)
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.0):
        """Calculate Sortino ratio (only considers downside risk)."""
        if len(self.returns) < 2:
            return 0.0
        
        # Calculate mean return
        mean_return = np.mean(self.returns)
        
        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in self.returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0.0001
        
        # Avoid division by zero
        if downside_deviation == 0:
            return 0.0
        
        # Calculate annualized Sortino ratio
        sortino = (mean_return - risk_free_rate) / downside_deviation * np.sqrt(252)
        
        return sortino
    
    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio (return / max drawdown)."""
        if self.max_drawdown == 0 or len(self.returns) < 2:
            return 0.0
        
        # Calculate annualized return
        total_return = (self.portfolio_value / self.initial_balance) - 1
        num_days = len(self.returns)
        annualized_return = (1 + total_return) ** (252 / num_days) - 1 if num_days > 0 else 0
        
        # Calculate Calmar ratio
        calmar = annualized_return / self.max_drawdown
        
        return calmar
    
    def _publish_progress(self):
        """Publish training progress to WebSocket channel."""
        try:
            # Create progress message
            progress_data = {
                "episode": self.episode_count,
                "step": self.current_step,
                "portfolio_value": self.portfolio_value,
                "return": (self.portfolio_value / self.initial_balance) - 1,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "total_trades": len(self.trades),
                "params": {param['name']: self.strategy.indicators[0].parameters.get(param['param'], 0) 
                           for param in self.action_parameters if param['indicator'] in [ind.name for ind in self.strategy.indicators]}
            }
            
            # Create async task to publish
            asyncio.create_task(MessagePublisher.publish(self.websocket_channel, progress_data))
        except Exception as e:
            logger.error(f"Error publishing progress: {str(e)}")
    
    def calculate_performance_metrics(self):
        """Calculate final performance metrics for the strategy."""
        # Calculate basic metrics
        total_return = (self.portfolio_value / self.initial_balance) - 1
        win_trades = [t for t in self.trades if t.get("profit_loss", 0) > 0]
        loss_trades = [t for t in self.trades if t.get("profit_loss", 0) <= 0]
        win_rate = len(win_trades) / len(self.trades) if self.trades else 0
        
        # Calculate average profit/loss
        avg_profit = np.mean([t.get("profit_loss", 0) for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t.get("profit_loss", 0) for t in loss_trades]) if loss_trades else 0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "best_parameters": self.best_params
        }
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics for the strategy."""
        returns = np.array(self.returns)
        
        # Calculate metrics
        total_return = (self.portfolio_value / 10000.0) - 1.0
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
        max_drawdown = 0.0
        win_rate = 0.0
        
        # Calculate drawdown
        if len(returns) > 0:
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            max_drawdown = np.max(drawdown)
        
        # Calculate win rate
        if len(self.trades) > 0:
            profitable_trades = sum(1 for trade in self.trades if trade["type"] == "sell" and trade["amount"] > 0)
            total_trades = sum(1 for trade in self.trades if trade["type"] == "sell")
            win_rate = profitable_trades / max(1, total_trades)
        
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "num_trades": len(self.trades)
        }


class RLService:
    """Service for Reinforcement Learning optimization."""
    
    def __init__(self):
        """Initialize the RL service."""
        self.models = {}  # Store trained models by strategy ID
    
    async def optimize_strategy(self, strategy: Strategy, historical_data: List[Dict[str, Any]]) -> Strategy:
        """Optimize a strategy using RL.
        
        Args:
            strategy: The strategy to optimize
            historical_data: Historical market data for training
            
        Returns:
            Optimized strategy
        """
        try:
            logger.info(f"Optimizing strategy {strategy.id} using RL")
            
            # Create environment
            def make_env():
                return TradingEnvironment(strategy, historical_data)
            
            env = DummyVecEnv([make_env])
            
            # Initialize PPO agent
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
        except Exception as e:
            logger.error(f"Error training model for strategy {strategy_id}: {str(e)}")
            if websocket_channel:
                await MessagePublisher.publish(websocket_channel, {
                    "type": "rl_training_error",
                    "strategy_id": strategy_id,
                    "error": str(e)
                })
    
    async def backtest_strategy(self, strategy: Strategy, historical_data: List[Dict[str, Any]]) -> BacktestResult:
        """Backtest a strategy against historical data with detailed metrics.
        
        Args:
            strategy: The strategy to backtest
            historical_data: Historical market data
            
        Returns:
            Backtest results
        """
        logger.info(f"Backtesting strategy {strategy.id}")
        
        # Create the trading environment with risk management
        env = TradingEnvironment(
            strategy=strategy, 
            historical_data=historical_data,
            risk_adjusted=True,  # Always use risk management for backtesting
            reward_function='returns'  # Use simple returns for backtesting
        )
        
        # Reset the environment
        obs, _ = env.reset()
        done = False
        
        # Run through the environment
        while not done:
            # If we have a trained model for this strategy, use it
            if strategy.id in self.models:
                action, _ = self.models[strategy.id].predict(obs, deterministic=True)
            else:
                # Otherwise use the strategy's current parameters
                action = np.array([param.get('default', 0) for param in env.action_parameters])
            
            obs, reward, done, _, info = env.step(action)
        
        # Calculate performance metrics
        metrics = env.calculate_performance_metrics()
        
        # Extract trade history for analysis
        trades = env.trades
        
        # Create detailed backtest result
        backtest_result = BacktestResult(
            strategy_id=strategy.id,
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            calmar_ratio=metrics["calmar_ratio"],
            max_drawdown=metrics["max_drawdown"],
            volatility=metrics["volatility"],
            win_rate=metrics["win_rate"],
            profit_loss_ratio=metrics["profit_loss_ratio"],
            total_trades=len(trades),
            trades=trades,
            optimized_parameters=metrics["best_parameters"]
        )
        
        return backtest_result
    
    async def get_optimization_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get the status of an ongoing optimization.
        
        Args:
            strategy_id: ID of the strategy being optimized
            
        Returns:
            Status information
        """
        if strategy_id not in self.training_tasks:
            return {"status": "not_found", "message": f"No optimization in progress for strategy {strategy_id}"}
        
        task = self.training_tasks[strategy_id]
        
        if task.done():
            if task.exception():
                return {"status": "error", "message": str(task.exception())}
            else:
                return {"status": "completed", "message": "Optimization completed successfully"}
        else:
            return {"status": "in_progress", "message": "Optimization is still running"}
    
    async def stop_optimization(self, strategy_id: str) -> Dict[str, Any]:
        """Stop an ongoing optimization.
        
        Args:
            strategy_id: ID of the strategy being optimized
            
        Returns:
            Status information
        """
        if strategy_id not in self.training_tasks:
            return {"status": "not_found", "message": f"No optimization in progress for strategy {strategy_id}"}
        
        task = self.training_tasks[strategy_id]
        
        if not task.done():
            task.cancel()
            return {"status": "stopped", "message": f"Optimization for strategy {strategy_id} has been stopped"}
        else:
            return {"status": "already_completed", "message": f"Optimization for strategy {strategy_id} already completed"}
            
            logger.info(f"Backtest complete with return: {metrics['total_return']:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Error backtesting strategy: {str(e)}")
            raise


# Create a singleton instance
rl_service = RLService()
