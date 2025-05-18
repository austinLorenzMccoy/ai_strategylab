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

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
from typing import Optional, Dict, Any
import uuid
import logging
import json
from app.services.websocket_service import WebSocketHandler, MessagePublisher
from app.services.sentiment_analysis_service import SentimentAnalysisService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize sentiment analysis service
sentiment_service = SentimentAnalysisService(websocket_channel="sentiment_updates")


@router.on_event("startup")
async def startup_sentiment_service():
    """Initialize the sentiment analysis service on startup."""
    await sentiment_service.initialize()


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


# Sentiment Analysis HTTP Endpoints
@router.get("/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """
    Get sentiment analysis for a specific cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "BTC")
        
    Returns:
        Sentiment analysis data for the specified cryptocurrency
    """
    sentiment = await sentiment_service.get_sentiment(symbol.upper())
    if not sentiment:
        raise HTTPException(status_code=404, detail=f"No sentiment data available for {symbol}")
    
    return sentiment


@router.get("/sentiment")
async def get_all_sentiments():
    """
    Get sentiment analysis for all cryptocurrencies.
    
    Returns:
        Dictionary of sentiment analysis data by cryptocurrency symbol
    """
    sentiments = await sentiment_service.get_all_sentiments()
    return {symbol: sentiment.dict() for symbol, sentiment in sentiments.items()}


@router.get("/sentiment/market/overview")
async def get_market_sentiment():
    """
    Get overall market sentiment.
    
    Returns:
        Market sentiment metrics including sentiment score, FUD level, and FOMO level
    """
    return await sentiment_service.get_market_sentiment()


@router.get("/sentiment/signals")
async def get_trading_signals():
    """
    Get trading signals based on sentiment analysis.
    
    Returns:
        Dictionary of trading signals by cryptocurrency symbol
    """
    return await sentiment_service.generate_trading_signals()
