"""
WebSocket Routes Module

This module defines the WebSocket endpoints for the AI Strategy Lab application.
It provides real-time data streaming capabilities through WebSocket connections,
serving as an alternative to Kafka for low-latency communication.

Endpoints:
- /ws/{channel}: WebSocket endpoint for real-time data streaming
  - Supports channels like "market_data", "strategy_signals", etc.
  - Handles client connections, disconnections, and message routing

This implementation allows for real-time data streaming between the backend and
frontend clients, as well as between different components of the application.
It eliminates the need for Kafka setup and maintenance while providing similar
functionality with lower latency and direct browser compatibility.

Example usage (JavaScript client):
    const socket = new WebSocket('ws://localhost:8000/ws/market_data');
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received:', data);
    };
    socket.send(JSON.stringify({action: 'subscribe'}));
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional
import uuid
import logging
from app.services.websocket_service import WebSocketHandler, MessagePublisher

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/{channel}")
async def websocket_endpoint(
    websocket: WebSocket, 
    channel: str,
    client_id: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time data streaming.
    
    Args:
        websocket: The WebSocket connection
        channel: The channel to subscribe to (e.g., "market_data", "strategy_updates")
        client_id: Optional client identifier
    """
    # Generate a client ID if not provided
    if not client_id:
        client_id = f"client-{uuid.uuid4()}"
    
    await WebSocketHandler.websocket_endpoint(websocket, client_id, channel)


# Example function to publish data to a WebSocket channel
async def publish_to_channel(channel: str, data: dict):
    """
    Publish data to a WebSocket channel.
    
    Args:
        channel: The channel to publish to
        data: The data to publish
    """
    await MessagePublisher.publish(channel, data)
