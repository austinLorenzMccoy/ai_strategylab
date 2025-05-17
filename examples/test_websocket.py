#!/usr/bin/env python
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """Test the WebSocket connection to the FastAPI server."""
    uri = "ws://localhost:8001/ws/market_data"
    
    logger.info(f"Connecting to WebSocket server at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server!")
            
            # Send a message to the server
            await websocket.send(json.dumps({"client": "test_client", "action": "subscribe"}))
            logger.info("Sent subscription message")
            
            # Receive a message
            response = await websocket.recv()
            logger.info(f"Received: {response}")
            
            # Keep connection open for a few seconds
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_websocket_connection())
