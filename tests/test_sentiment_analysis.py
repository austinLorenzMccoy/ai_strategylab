"""
Test script for the Sentiment Analysis Service.

This script demonstrates how to use the sentiment analysis service
to analyze cryptocurrency sentiment from various sources and generate
trading signals based on the sentiment data.
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.sentiment_analysis_service import SentimentAnalysisService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_sentiment_analysis():
    """Test the sentiment analysis service."""
    logger.info("Initializing sentiment analysis service...")
    
    # Initialize the service
    service = SentimentAnalysisService(websocket_channel="sentiment_updates")
    await service.initialize()
    
    # Wait for initial data collection
    logger.info("Waiting for initial data collection (30 seconds)...")
    await asyncio.sleep(30)
    
    # Get sentiment for Bitcoin
    btc_sentiment = await service.get_sentiment("BTC")
    if btc_sentiment:
        logger.info(f"Bitcoin sentiment: {btc_sentiment.sentiment_label} ({btc_sentiment.sentiment_score:.2f})")
        logger.info(f"Volume: {btc_sentiment.volume} mentions")
        logger.info(f"Trend: {btc_sentiment.trend:.2f}")
        logger.info(f"Sources: {json.dumps(btc_sentiment.sources, indent=2)}")
    else:
        logger.info("No sentiment data available for Bitcoin yet")
    
    # Get market sentiment
    market_sentiment = await service.get_market_sentiment()
    logger.info(f"Market sentiment: {market_sentiment['sentiment_label']} ({market_sentiment['sentiment_score']:.2f})")
    logger.info(f"FUD level: {market_sentiment['fud_level']:.2f}")
    logger.info(f"FOMO level: {market_sentiment['fomo_level']:.2f}")
    
    # Generate trading signals
    signals = await service.generate_trading_signals()
    logger.info(f"Generated {len(signals)} trading signals:")
    for symbol, signal in signals.items():
        logger.info(f"  {symbol}: {signal['signal']} (strength: {signal['strength']:.2f}) - {signal['reason']}")
    
    # Get all sentiments
    all_sentiments = await service.get_all_sentiments()
    logger.info(f"Sentiment data available for {len(all_sentiments)} cryptocurrencies:")
    for symbol in all_sentiments:
        logger.info(f"  {symbol}")
    
    logger.info("Sentiment analysis test completed")


if __name__ == "__main__":
    asyncio.run(test_sentiment_analysis())
