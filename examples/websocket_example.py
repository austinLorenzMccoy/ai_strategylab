import asyncio
import json
import logging
import sys
import os
import random
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.websocket_agent import MarketDataAgent, StrategyAgent
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def simulate_market_data(interval=1.0):
    """
    Simulate market data for testing.
    
    Args:
        interval: Time interval between data points in seconds
    """
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    market_data_agent = MarketDataAgent()
    
    logger.info("Starting market data simulation...")
    
    while True:
        for symbol in symbols:
            # Generate random price data
            price = random.uniform(
                10000 if symbol == "BTC-USDT" else (1000 if symbol == "ETH-USDT" else 100),
                11000 if symbol == "BTC-USDT" else (1100 if symbol == "ETH-USDT" else 110)
            )
            
            # Create market data
            market_data = {
                "symbol": symbol,
                "price": price,
                "volume": random.uniform(10, 100),
                "timestamp": time.time()
            }
            
            # Process and publish market data
            await market_data_agent.run(market_data)
            logger.info(f"Published market data for {symbol}: {price:.2f}")
        
        # Wait for the next interval
        await asyncio.sleep(interval)


async def run_strategy_agent():
    """Run a strategy agent that listens for market data and generates signals."""
    strategy_agent = StrategyAgent(strategy_id="example-strategy-1")
    
    logger.info("Starting strategy agent...")
    
    # In a real application, the strategy agent would subscribe to market data
    # through a WebSocket connection. For this example, we'll simulate it by
    # creating random market data.
    
    while True:
        # Generate random market data
        symbol = random.choice(["BTC-USDT", "ETH-USDT", "SOL-USDT"])
        price = random.uniform(
            10000 if symbol == "BTC-USDT" else (1000 if symbol == "ETH-USDT" else 100),
            11000 if symbol == "BTC-USDT" else (1100 if symbol == "ETH-USDT" else 110)
        )
        
        market_data = {
            "symbol": symbol,
            "price": price,
            "volume": random.uniform(10, 100),
            "timestamp": time.time()
        }
        
        # Process market data and generate signals
        result = await strategy_agent.run(market_data)
        if result.get("signal"):
            logger.info(f"Strategy signal for {symbol}: {result['signal']} at {price:.2f}")
        
        # Wait before processing next data point
        await asyncio.sleep(2.0)


async def websocket_client():
    """
    Example WebSocket client that connects to our server and receives messages.
    In a real application, this could be a frontend or another service.
    """
    uri = "ws://localhost:8000/ws/market_data"
    
    logger.info(f"Connecting to WebSocket server at {uri}...")
    
    async with websockets.connect(uri) as websocket:
        logger.info("Connected to WebSocket server!")
        
        # Send a message to the server
        await websocket.send(json.dumps({"client": "example_client", "action": "subscribe"}))
        
        # Receive messages
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                logger.info(f"Received: {data}")
            except Exception as e:
                logger.error(f"Error receiving message: {str(e)}")
                break


async def main():
    """Run the example."""
    # Create tasks for our components
    market_data_task = asyncio.create_task(simulate_market_data())
    strategy_task = asyncio.create_task(run_strategy_agent())
    
    # In a real application, you would run a WebSocket client in a separate process
    # or on a different machine. For this example, we'll run it in a separate task.
    client_task = asyncio.create_task(websocket_client())
    
    # Wait for all tasks to complete (they won't in this example, as they run indefinitely)
    await asyncio.gather(market_data_task, strategy_task, client_task)


if __name__ == "__main__":
    try:
        # Make sure the FastAPI server is running before executing this script
        print("Make sure the FastAPI server is running (uvicorn app.main:app --reload)")
        print("Press Enter to continue or Ctrl+C to exit...")
        input()
        
        # Run the example
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Example stopped by user")
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
