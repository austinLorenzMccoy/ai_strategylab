#!/usr/bin/env python
import asyncio
import websockets
import json
import logging
import random
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Store connected clients
connected_clients = set()

async def handle_client(websocket):
    """Handle a client connection."""
    # Register client
    connected_clients.add(websocket)
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected")
    
    try:
        # Handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Received from client {client_id}: {data}")
                
                # Echo back the message
                await websocket.send(json.dumps({
                    "type": "echo",
                    "data": data,
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

async def broadcast_market_data():
    """Broadcast simulated market data to all connected clients."""
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    
    while True:
        if connected_clients:
            # Generate random market data
            symbol = random.choice(symbols)
            price = random.uniform(
                10000 if symbol == "BTC-USDT" else (1000 if symbol == "ETH-USDT" else 100),
                11000 if symbol == "BTC-USDT" else (1100 if symbol == "ETH-USDT" else 110)
            )
            
            market_data = {
                "type": "market_data",
                "symbol": symbol,
                "price": price,
                "volume": random.uniform(10, 100),
                "timestamp": time.time()
            }
            
            # Broadcast to all clients
            websockets_tasks = [
                client.send(json.dumps(market_data))
                for client in connected_clients
            ]
            
            if websockets_tasks:
                await asyncio.gather(*websockets_tasks, return_exceptions=True)
                logger.info(f"Broadcast market data for {symbol}: {price:.2f} to {len(connected_clients)} clients")
        
        # Wait before sending next update
        await asyncio.sleep(2)

async def main():
    # Start the WebSocket server
    server = await websockets.serve(handle_client, "localhost", 8765)
    logger.info("WebSocket server started on ws://localhost:8765")
    
    # Start broadcasting market data
    broadcast_task = asyncio.create_task(broadcast_market_data())
    
    # Keep the server running
    await server.wait_closed()
    broadcast_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
