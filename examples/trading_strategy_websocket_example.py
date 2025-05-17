#!/usr/bin/env python
"""
Trading Strategy WebSocket Example

This example demonstrates how to use the WebSocket-based real-time data streaming
for implementing a simple trading strategy in the AI Strategy Lab.

The example includes:
1. A market data source that simulates real-time price updates
2. A trading strategy that processes the market data and generates signals
3. A WebSocket client that receives and displays the trading signals

Run this example after starting the FastAPI server with:
    uvicorn app.main:app --reload
"""

import asyncio
import json
import logging
import random
import time
import sys
import os
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.websocket_agent import MarketDataAgent, StrategyAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleMovingAverageStrategy:
    """
    A simple moving average crossover trading strategy.
    
    This strategy generates buy signals when the short-term moving average
    crosses above the long-term moving average, and sell signals when the
    short-term moving average crosses below the long-term moving average.
    """
    
    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20):
        """
        Initialize the strategy.
        
        Args:
            symbol: The trading symbol to apply the strategy to
            short_window: The window size for the short-term moving average
            long_window: The window size for the long-term moving average
        """
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.prices: List[float] = []
        self.last_signal: Optional[str] = None
    
    def update(self, price: float) -> Optional[Dict[str, Any]]:
        """
        Update the strategy with a new price and generate a signal if applicable.
        
        Args:
            price: The new price
            
        Returns:
            A trading signal dictionary or None if no signal is generated
        """
        # Add the new price to the price history
        self.prices.append(price)
        
        # We need at least long_window prices to calculate the moving averages
        if len(self.prices) < self.long_window:
            return None
        
        # Calculate the moving averages
        short_ma = sum(self.prices[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices[-self.long_window:]) / self.long_window
        
        # Generate signals based on moving average crossover
        signal = None
        if short_ma > long_ma and (self.last_signal is None or self.last_signal == "SELL"):
            signal = "BUY"
            self.last_signal = signal
        elif short_ma < long_ma and (self.last_signal is None or self.last_signal == "BUY"):
            signal = "SELL"
            self.last_signal = signal
        
        # If we have a signal, return a signal dictionary
        if signal:
            return {
                "symbol": self.symbol,
                "price": price,
                "signal": signal,
                "short_ma": short_ma,
                "long_ma": long_ma,
                "timestamp": time.time()
            }
        
        return None


class TradingStrategyAgent(StrategyAgent):
    """
    A trading strategy agent that uses the SimpleMovingAverageStrategy.
    """
    
    def __init__(self, strategy_id: str, symbols: List[str]):
        """
        Initialize the trading strategy agent.
        
        Args:
            strategy_id: The ID of the strategy
            symbols: The list of symbols to trade
        """
        super().__init__(name=f"TradingStrategy_{strategy_id}", strategy_id=strategy_id)
        self.strategies = {
            symbol: SimpleMovingAverageStrategy(symbol)
            for symbol in symbols
        }
    
    async def _process_data(self, input_data: Any) -> Dict[str, Any]:
        """
        Process market data and generate trading signals.
        
        Args:
            input_data: Market data to process
            
        Returns:
            Trading signals or processed data
        """
        if not isinstance(input_data, dict):
            return {"error": "Invalid input data format"}
        
        symbol = input_data.get("symbol")
        price = input_data.get("price")
        
        if not symbol or not price or symbol not in self.strategies:
            return input_data
        
        # Update the strategy with the new price
        signal_data = self.strategies[symbol].update(price)
        
        # If we have a signal, return it
        if signal_data:
            logger.info(f"Generated {signal_data['signal']} signal for {symbol} at {price:.2f}")
            return {
                "strategy_id": self.strategy_id,
                "symbol": symbol,
                "price": price,
                "signal": signal_data["signal"],
                "short_ma": signal_data["short_ma"],
                "long_ma": signal_data["long_ma"],
                "timestamp": time.time()
            }
        
        # Otherwise, return the input data
        return input_data


async def simulate_market_data(interval: float = 1.0, volatility: float = 0.002):
    """
    Simulate market data for testing.
    
    Args:
        interval: Time interval between data points in seconds
        volatility: Price volatility factor
    """
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    base_prices = {
        "BTC-USDT": 50000.0,
        "ETH-USDT": 3000.0,
        "SOL-USDT": 100.0
    }
    current_prices = base_prices.copy()
    
    market_data_agent = MarketDataAgent()
    
    logger.info("Starting market data simulation...")
    
    while True:
        for symbol in symbols:
            # Generate random price movement
            price_change = current_prices[symbol] * volatility * (random.random() * 2 - 1)
            current_prices[symbol] += price_change
            
            # Create market data
            market_data = {
                "symbol": symbol,
                "price": current_prices[symbol],
                "volume": random.uniform(10, 100),
                "timestamp": time.time()
            }
            
            # Process and publish market data
            await market_data_agent.run(market_data)
            logger.info(f"Published market data for {symbol}: {current_prices[symbol]:.2f}")
        
        # Wait for the next interval
        await asyncio.sleep(interval)


async def run_trading_strategy():
    """
    Run a trading strategy agent that listens for market data and generates signals.
    """
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    strategy_agent = TradingStrategyAgent(strategy_id="sma-crossover", symbols=symbols)
    
    logger.info("Starting trading strategy agent...")
    
    # In a real application, the strategy agent would subscribe to market data
    # through a WebSocket connection. For this example, we'll simulate it by
    # processing market data directly.
    
    base_prices = {
        "BTC-USDT": 50000.0,
        "ETH-USDT": 3000.0,
        "SOL-USDT": 100.0
    }
    current_prices = base_prices.copy()
    
    # Initialize with some historical data
    for symbol in symbols:
        for _ in range(30):  # Generate 30 historical data points
            price_change = current_prices[symbol] * 0.002 * (random.random() * 2 - 1)
            current_prices[symbol] += price_change
            
            market_data = {
                "symbol": symbol,
                "price": current_prices[symbol],
                "timestamp": time.time() - (30 - _) * 60  # Backdate the timestamps
            }
            
            await strategy_agent.run(market_data)
    
    # Now process real-time data
    while True:
        for symbol in symbols:
            # Generate random price movement
            price_change = current_prices[symbol] * 0.002 * (random.random() * 2 - 1)
            current_prices[symbol] += price_change
            
            market_data = {
                "symbol": symbol,
                "price": current_prices[symbol],
                "volume": random.uniform(10, 100),
                "timestamp": time.time()
            }
            
            # Process market data and generate signals
            result = await strategy_agent.run(market_data)
            if result.get("signal"):
                logger.info(
                    f"Strategy signal for {symbol}: {result['signal']} at {current_prices[symbol]:.2f} "
                    f"(Short MA: {result.get('short_ma', 0):.2f}, Long MA: {result.get('long_ma', 0):.2f})"
                )
        
        # Wait before processing next data point
        await asyncio.sleep(2.0)


async def main():
    """Run the example."""
    # Create tasks for our components
    market_data_task = asyncio.create_task(simulate_market_data())
    strategy_task = asyncio.create_task(run_trading_strategy())
    
    # Wait for all tasks to complete (they won't in this example, as they run indefinitely)
    await asyncio.gather(market_data_task, strategy_task)


if __name__ == "__main__":
    try:
        # Make sure the FastAPI server is running before executing this script
        print("This example demonstrates a trading strategy using WebSocket-based real-time data streaming.")
        print("Make sure the FastAPI server is running (uvicorn app.main:app --reload)")
        print("Press Enter to continue or Ctrl+C to exit...")
        input()
        
        # Run the example
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Example stopped by user")
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
