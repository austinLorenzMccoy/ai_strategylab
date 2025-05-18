"""
WebSocket Service Module

This module provides a WebSocket-based real-time data streaming implementation as an
alternative to Kafka. It handles connection management, message broadcasting, and
channel-based publish/subscribe functionality.

Key components:
- ConnectionManager: Manages WebSocket connections and message broadcasting
- MessagePublisher: Publishes messages to WebSocket channels
- WebSocketHandler: Handles WebSocket connections and messages

This implementation allows for real-time data streaming between agents and with
frontend clients without the complexity of Kafka setup and maintenance.

Example usage:
    # In a FastAPI route
    @router.websocket("/ws/{channel}")
    async def websocket_endpoint(websocket: WebSocket, channel: str, client_id: str):
        await WebSocketHandler.websocket_endpoint(websocket, client_id, channel)
        
    # Publishing messages
    await MessagePublisher.publish("market_data", {"symbol": "BTC-USDT", "price": 50000})
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.background_tasks = set()
    
    async def connect(self, websocket: WebSocket, client_id: str, channel: str):
        """Connect a client to a specific channel."""
        await websocket.accept()
        
        if channel not in self.active_connections:
            self.active_connections[channel] = []
            self.message_queues[channel] = asyncio.Queue()
            # Start background task for this channel
            task = asyncio.create_task(self._process_queue(channel))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        self.active_connections[channel].append(websocket)
        logger.info(f"Client {client_id} connected to channel {channel}")
    
    def disconnect(self, websocket: WebSocket, client_id: str, channel: str):
        """Disconnect a client from a channel."""
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)
                logger.info(f"Client {client_id} disconnected from channel {channel}")
            
            # If no more connections in this channel, clean up
            if not self.active_connections[channel]:
                self.active_connections.pop(channel, None)
                # Don't remove the queue yet, as there might be pending messages
    
    async def broadcast(self, message: Any, channel: str):
        """Add a message to the channel's queue for broadcasting."""
        if channel in self.message_queues:
            await self.message_queues[channel].put(message)
            logger.debug(f"Message added to queue for channel {channel}")
        else:
            logger.warning(f"Attempted to broadcast to non-existent channel: {channel}")
    
    async def _process_queue(self, channel: str):
        """Process messages in the queue and send them to all connected clients."""
        try:
            while channel in self.active_connections:
                # Wait for a message
                message = await self.message_queues[channel].get()
                
                # Send to all connected clients
                disconnected = []
                for connection in self.active_connections.get(channel, []):
                    try:
                        if isinstance(message, dict) or isinstance(message, list):
                            await connection.send_json(message)
                        else:
                            await connection.send_text(str(message))
                    except Exception as e:
                        logger.error(f"Error sending message to client: {str(e)}")
                        disconnected.append(connection)
                
                # Remove disconnected clients
                for conn in disconnected:
                    if channel in self.active_connections and conn in self.active_connections[channel]:
                        self.active_connections[channel].remove(conn)
                
                # Mark task as done
                self.message_queues[channel].task_done()
        except asyncio.CancelledError:
            logger.info(f"Queue processor for channel {channel} cancelled")
        except Exception as e:
            logger.error(f"Error in queue processor for channel {channel}: {str(e)}")


# Global connection manager
manager = ConnectionManager()


class MessagePublisher:
    """Publishes messages to WebSocket channels."""
    
    @staticmethod
    async def publish(channel: str, message: Any):
        """Publish a message to a channel."""
        await manager.broadcast(message, channel)


class WebSocketHandler:
    """Handles WebSocket connections and messages."""
    
    @staticmethod
    async def websocket_endpoint(websocket: WebSocket, client_id: str, channel: str):
        """WebSocket endpoint for a specific channel."""
        try:
            await manager.connect(websocket, client_id, channel)
            
            # Keep the connection alive and handle incoming messages
            while True:
                data = await websocket.receive_text()
                try:
                    # Echo back received messages (optional)
                    message = json.loads(data)
                    # You can process incoming messages here if needed
                    await websocket.send_json({"status": "received", "data": message})
                except json.JSONDecodeError:
                    await websocket.send_text(f"Received: {data}")
                
        except WebSocketDisconnect:
            manager.disconnect(websocket, client_id, channel)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            manager.disconnect(websocket, client_id, channel)
