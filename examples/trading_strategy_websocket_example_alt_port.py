#!/usr/bin/env python
"""
Trading Strategy WebSocket Example (Alternative Port)

This example demonstrates how to use the WebSocket-based real-time data streaming
for implementing a simple trading strategy in the AI Strategy Lab.

This version uses port 8767 to avoid port conflicts.
"""

import asyncio
import json
import logging
import random
import time
import sys
import os
import websockets
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Store connected clients
connected_clients = set()

# Store price history for each symbol
price_history = {
    "BTC-USDT": [],
    "ETH-USDT": [],
    "SOL-USDT": []
}

# Base prices for the symbols
base_prices = {
    "BTC-USDT": 50000.0,
    "ETH-USDT": 3000.0,
    "SOL-USDT": 100.0
}

# Current prices
current_prices = base_prices.copy()

# Trading signals
trading_signals = []


class SimpleMovingAverageStrategy:
    """
    A simple moving average crossover trading strategy.
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


async def handle_client(websocket):
    """Handle a client connection."""
    # Register client
    connected_clients.add(websocket)
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected")
    
    # Send initial data to the client
    await websocket.send(json.dumps({
        "type": "init",
        "price_history": price_history,
        "trading_signals": trading_signals,
        "timestamp": time.time()
    }))
    
    try:
        # Handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Received from client {client_id}: {data}")
                
                # Handle client messages (e.g., subscribe to specific symbols)
                if data.get("action") == "subscribe":
                    await websocket.send(json.dumps({
                        "type": "subscription",
                        "status": "success",
                        "message": f"Subscribed to all available symbols",
                        "timestamp": time.time()
                    }))
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON from client {client_id}: {message}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": time.time()
                }))
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")
    finally:
        # Unregister client
        connected_clients.remove(websocket)


async def simulate_market_data():
    """Simulate market data and update trading strategies."""
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    strategies = {
        symbol: SimpleMovingAverageStrategy(symbol)
        for symbol in symbols
    }
    
    # Initialize with some historical data
    timestamp = time.time() - 60 * 30  # Start 30 minutes ago
    for _ in range(30):
        for symbol in symbols:
            price_change = current_prices[symbol] * 0.002 * (random.random() * 2 - 1)
            current_prices[symbol] += price_change
            
            # Add to price history
            price_history[symbol].append({
                "price": current_prices[symbol],
                "timestamp": timestamp
            })
            
            # Update strategy
            strategies[symbol].update(current_prices[symbol])
        
        timestamp += 60  # Increment by 1 minute
    
    # Now generate real-time data
    while True:
        current_timestamp = time.time()
        
        for symbol in symbols:
            # Generate random price movement
            price_change = current_prices[symbol] * 0.002 * (random.random() * 2 - 1)
            current_prices[symbol] += price_change
            
            # Add to price history (keep only the last 100 data points)
            price_history[symbol].append({
                "price": current_prices[symbol],
                "timestamp": current_timestamp
            })
            if len(price_history[symbol]) > 100:
                price_history[symbol].pop(0)
            
            # Update strategy and check for signals
            signal_data = strategies[symbol].update(current_prices[symbol])
            if signal_data:
                logger.info(f"Generated {signal_data['signal']} signal for {symbol} at {current_prices[symbol]:.2f}")
                
                # Add to trading signals
                trading_signals.append({
                    "symbol": symbol,
                    "price": current_prices[symbol],
                    "signal": signal_data["signal"],
                    "short_ma": signal_data["short_ma"],
                    "long_ma": signal_data["long_ma"],
                    "timestamp": current_timestamp
                })
                
                # Keep only the last 20 signals
                if len(trading_signals) > 20:
                    trading_signals.pop(0)
                
                # Broadcast signal to all clients
                signal_message = {
                    "type": "signal",
                    "symbol": symbol,
                    "price": current_prices[symbol],
                    "signal": signal_data["signal"],
                    "short_ma": signal_data["short_ma"],
                    "long_ma": signal_data["long_ma"],
                    "timestamp": current_timestamp
                }
                
                if connected_clients:
                    signal_tasks = [
                        client.send(json.dumps(signal_message))
                        for client in connected_clients
                    ]
                    
                    if signal_tasks:
                        await asyncio.gather(*signal_tasks, return_exceptions=True)
            
            # Create market data message
            market_data = {
                "type": "market_data",
                "symbol": symbol,
                "price": current_prices[symbol],
                "volume": random.uniform(10, 100),
                "timestamp": current_timestamp
            }
            
            # Broadcast to all clients
            if connected_clients:
                websockets_tasks = [
                    client.send(json.dumps(market_data))
                    for client in connected_clients
                ]
                
                if websockets_tasks:
                    await asyncio.gather(*websockets_tasks, return_exceptions=True)
                    logger.debug(f"Broadcast market data for {symbol}: {current_prices[symbol]:.2f} to {len(connected_clients)} clients")
        
        # Wait before sending next update
        await asyncio.sleep(1)


async def main():
    """Run the trading strategy example."""
    # Start the WebSocket server
    server = await websockets.serve(handle_client, "localhost", 8767)
    logger.info("WebSocket server started on ws://localhost:8767")
    
    # Print connection instructions
    print("\nTo connect to this WebSocket server from JavaScript:")
    print("const socket = new WebSocket('ws://localhost:8767');")
    print("\nTo connect from Python:")
    print("import websockets")
    print("async with websockets.connect('ws://localhost:8767') as websocket:")
    print("    # Send and receive messages here")
    
    # Start simulating market data
    market_data_task = asyncio.create_task(simulate_market_data())
    
    # Keep the server running
    await server.wait_closed()


if __name__ == "__main__":
    try:
        print("Starting Trading Strategy WebSocket Example...")
        print("This example demonstrates a trading strategy using WebSocket-based real-time data streaming.")
        
        # Run the example
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Example stopped by user")
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
