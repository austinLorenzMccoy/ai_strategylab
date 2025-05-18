#!/usr/bin/env python
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def connect_and_receive():
    """Connect to the WebSocket server and receive messages."""
    uri = "ws://localhost:8765"
    
    logger.info(f"Connecting to WebSocket server at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server!")
            
            # Send a subscription message
            await websocket.send(json.dumps({
                "client": "python_client",
                "action": "subscribe"
            }))
            logger.info("Sent subscription message")
            
            # Receive messages in a loop
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    logger.info(f"Received: {data}")
                except Exception as e:
                    logger.error(f"Error receiving message: {str(e)}")
                    break
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(connect_and_receive())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
