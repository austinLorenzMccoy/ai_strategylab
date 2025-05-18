"""
Sentiment Analysis Service Module

This module provides a service for analyzing sentiment from cryptocurrency news
and social media sources to generate trading signals. It uses NLP techniques to
process text data and extract sentiment scores that can be used to inform
trading decisions.

Key features:
- Analyze sentiment from news articles and social media posts
- Track sentiment trends over time for different cryptocurrencies
- Generate trading signals based on sentiment shifts
- Integrate with the WebSocket system for real-time updates
- Filter out high FUD (Fear, Uncertainty, Doubt) periods

This implementation helps traders avoid making decisions during periods of
extreme market sentiment and identify potential trading opportunities based
on sentiment analysis.
"""

import logging
import asyncio
import json
import time
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

# NLP libraries
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Project imports
from app.services.websocket_service import MessagePublisher
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SentimentSource(BaseModel):
    """Model for a sentiment data source."""
    name: str
    type: str  # "news", "social", "forum", etc.
    url: Optional[str] = None
    api_key: Optional[str] = None
    weight: float = 1.0  # Weight in the overall sentiment calculation
    update_frequency: int = 3600  # Update frequency in seconds


class SentimentData(BaseModel):
    """Model for sentiment analysis data."""
    source: str
    text: str
    title: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    symbols: List[str] = []  # Cryptocurrency symbols mentioned


class SentimentResult(BaseModel):
    """Model for sentiment analysis result."""
    data_id: str
    source: str
    text_snippet: str
    symbols: List[str]
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str  # "positive", "negative", "neutral"
    confidence: float
    timestamp: datetime
    relevance_score: float  # 0.0 to 1.0
    emotion_scores: Optional[Dict[str, float]] = None


class AggregateSentiment(BaseModel):
    """Model for aggregate sentiment for a cryptocurrency."""
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str  # "positive", "negative", "neutral"
    volume: int  # Number of mentions
    trend: float  # Change in sentiment over time
    sources: Dict[str, float]  # Sentiment by source
    timestamp: datetime
    historical: List[Tuple[datetime, float]] = []  # Historical sentiment


class SentimentAnalysisService:
    """Service for analyzing sentiment from cryptocurrency news and social media."""
    
    def __init__(self, websocket_channel: Optional[str] = None):
        """Initialize the sentiment analysis service.
        
        Args:
            websocket_channel: Optional channel for publishing sentiment updates
        """
        self.websocket_channel = websocket_channel
        self.sources = []  # Sentiment data sources
        self.sentiment_data = []  # Raw sentiment data
        self.sentiment_results = []  # Processed sentiment results
        self.aggregate_sentiments = {}  # Aggregate sentiment by symbol
        self.model = None  # Sentiment analysis model
        self.tokenizer = None  # Tokenizer for the model
        self.sentiment_pipeline = None  # Sentiment analysis pipeline
        self.update_task = None  # Background task for updating sentiment
        self.initialized = False  # Whether the service is initialized
        
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    async def initialize(self):
        """Initialize the sentiment analysis service."""
        if self.initialized:
            return
        
        logger.info("Initializing sentiment analysis service")
        
        # Load default sources
        self._load_default_sources()
        
        # Initialize sentiment analysis model
        await self._initialize_model()
        
        # Start background update task
        self.update_task = asyncio.create_task(self._background_update())
        
        self.initialized = True
        logger.info("Sentiment analysis service initialized")
    
    def _load_default_sources(self):
        """Load default sentiment data sources."""
        default_sources = []
        
        # Get API keys from settings, with fallbacks
        cryptocompare_api_key = getattr(settings, "CRYPTOCOMPARE_API_KEY", None)
        twitter_api_key = getattr(settings, "TWITTER_API_KEY", None)
        
        # Add CryptoCompare News source
        default_sources.append(
            SentimentSource(
                name="CryptoCompare News",
                type="news",
                url="https://min-api.cryptocompare.com/data/v2/news/",
                api_key=cryptocompare_api_key,
                weight=1.0,
                update_frequency=3600  # Every hour
            )
        )
        
        # Add Twitter Crypto source
        default_sources.append(
            SentimentSource(
                name="Twitter Crypto",
                type="social",
                url="https://api.twitter.com/2/tweets/search/recent",
                api_key=twitter_api_key,
                weight=0.8,
                update_frequency=1800  # Every 30 minutes
            )
        )
        
        # Add Reddit Crypto source
        default_sources.append(
            SentimentSource(
                name="Reddit Crypto",
                type="forum",
                url="https://www.reddit.com/r/CryptoCurrency/new.json",
                weight=0.7,
                update_frequency=3600  # Every hour
            )
        )
        
        self.sources = default_sources
        logger.info(f"Loaded {len(default_sources)} sentiment data sources")
        
        # Log warning if API keys are missing
        if not cryptocompare_api_key:
            logger.warning("CryptoCompare API key not found in settings. Using mock data for testing.")
        if not twitter_api_key:
            logger.warning("Twitter API key not found in settings. Using mock data for testing.")
    
    async def _initialize_model(self):
        """Initialize the sentiment analysis model."""
        try:
            # For testing purposes, we'll use a simple rule-based approach first
            # This avoids downloading large models during testing
            logger.info("Using simplified sentiment analysis for testing")
            self.use_simplified_sentiment = True
            self.sentiment_pipeline = None
            
            # In a production environment, you would use a proper NLP model:
            # Uncomment the following code for production use
            """
            # Load model and tokenizer
            model_name = "finiteautomata/bertweet-base-sentiment-analysis"
            
            logger.info(f"Loading sentiment analysis model: {model_name}")
            
            # Initialize the sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name
            )
            
            self.use_simplified_sentiment = False
            logger.info("Sentiment analysis model loaded successfully")
            """
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analysis model: {str(e)}")
            # Fallback to a simpler approach
            logger.info("Falling back to simplified sentiment analysis")
            self.use_simplified_sentiment = True
            self.sentiment_pipeline = None
    
    async def _background_update(self):
        """Background task for updating sentiment data."""
        while True:
            try:
                logger.info("Updating sentiment data")
                
                # Fetch and analyze sentiment from all sources
                for source in self.sources:
                    try:
                        # Check if it's time to update this source
                        # For now, update all sources
                        await self._fetch_sentiment_data(source)
                    except Exception as e:
                        logger.error(f"Error updating sentiment from {source.name}: {str(e)}")
                
                # Aggregate sentiment data
                await self._aggregate_sentiment()
                
                # Publish updates
                if self.websocket_channel:
                    await self._publish_sentiment_updates()
                
                logger.info("Sentiment data updated successfully")
                
            except Exception as e:
                logger.error(f"Error in sentiment update task: {str(e)}")
            
            # Sleep until next update
            await asyncio.sleep(60)  # Check every minute
    
    async def _fetch_sentiment_data(self, source: SentimentSource) -> List[SentimentData]:
        """Fetch sentiment data from a source.
        
        Args:
            source: Source to fetch data from
            
        Returns:
            List of sentiment data items
        """
        logger.info(f"Fetching sentiment data from {source.name}")
        
        data_items = []
        
        try:
            if source.type == "news" and source.name == "CryptoCompare News":
                data_items = await self._fetch_cryptocompare_news(source)
            elif source.type == "social" and source.name == "Twitter Crypto":
                data_items = await self._fetch_twitter_data(source)
            elif source.type == "forum" and source.name == "Reddit Crypto":
                data_items = await self._fetch_reddit_data(source)
            else:
                logger.warning(f"Unknown source type: {source.type} for {source.name}")
            
            # Process the fetched data
            if data_items:
                # Analyze sentiment for each item
                for item in data_items:
                    result = await self._analyze_sentiment(item)
                    if result:
                        self.sentiment_results.append(result)
                
                # Keep only the most recent 1000 results
                if len(self.sentiment_results) > 1000:
                    self.sentiment_results = self.sentiment_results[-1000:]
            
            return data_items
            
        except Exception as e:
            logger.error(f"Error fetching data from {source.name}: {str(e)}")
            return []
    
    async def _fetch_cryptocompare_news(self, source: SentimentSource) -> List[SentimentData]:
        """Fetch news from CryptoCompare API.
        
        Args:
            source: Source configuration
            
        Returns:
            List of sentiment data items
        """
        try:
            import aiohttp
            
            # Prepare URL with API key if available
            url = source.url
            if source.api_key:
                url += f"?api_key={source.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = data.get("Data", [])
                        
                        result = []
                        for item in news_items:
                            # Extract cryptocurrency symbols from tags
                            symbols = [tag.upper() for tag in item.get("tags", "").split("|")
                                     if tag.upper() in ["BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "XLM"]]
                            
                            # Create sentiment data item
                            sentiment_item = SentimentData(
                                source=source.name,
                                text=item.get("body", ""),
                                title=item.get("title", ""),
                                url=item.get("url", ""),
                                published_at=datetime.fromtimestamp(item.get("published_on", 0)),
                                author=item.get("source", ""),
                                symbols=symbols
                            )
                            
                            result.append(sentiment_item)
                        
                        return result
                    else:
                        logger.error(f"Error fetching CryptoCompare news: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error in _fetch_cryptocompare_news: {str(e)}")
            return []
    
    async def _fetch_twitter_data(self, source: SentimentSource) -> List[SentimentData]:
        """Fetch data from Twitter API.
        
        Args:
            source: Source configuration
            
        Returns:
            List of sentiment data items
        """
        try:
            import aiohttp
            
            # This is a simplified implementation
            # In a real implementation, you would use the Twitter API
            # For now, we'll return some mock data
            
            # Mock data for demonstration
            mock_tweets = [
                {
                    "text": "Bitcoin is looking bullish today! #BTC #crypto",
                    "created_at": datetime.now() - timedelta(hours=1),
                    "user": {"screen_name": "crypto_bull"}
                },
                {
                    "text": "Ethereum gas fees are too high again. Not sustainable for DeFi. #ETH",
                    "created_at": datetime.now() - timedelta(hours=2),
                    "user": {"screen_name": "defi_user"}
                },
                {
                    "text": "XRP lawsuit developments looking positive! #XRP #crypto",
                    "created_at": datetime.now() - timedelta(hours=3),
                    "user": {"screen_name": "xrp_fan"}
                }
            ]
            
            result = []
            for tweet in mock_tweets:
                # Extract cryptocurrency symbols from hashtags
                text = tweet.get("text", "")
                hashtags = re.findall(r'#(\w+)', text)
                symbols = [tag.upper() for tag in hashtags 
                         if tag.upper() in ["BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "XLM"]]
                
                # If no symbols found in hashtags, try to find them in text
                if not symbols:
                    for symbol in ["BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "XLM"]:
                        if symbol in text or symbol.lower() in text:
                            symbols.append(symbol)
                
                # Create sentiment data item
                sentiment_item = SentimentData(
                    source=source.name,
                    text=text,
                    published_at=tweet.get("created_at"),
                    author=tweet.get("user", {}).get("screen_name", ""),
                    symbols=symbols
                )
                
                result.append(sentiment_item)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _fetch_twitter_data: {str(e)}")
            return []
    
    async def _fetch_reddit_data(self, source: SentimentSource) -> List[SentimentData]:
        """Fetch data from Reddit.
        
        Args:
            source: Source configuration
            
        Returns:
            List of sentiment data items
        """
        try:
            import aiohttp
            
            # This is a simplified implementation
            # In a real implementation, you would use the Reddit API
            # For now, we'll return some mock data
            
            # Mock data for demonstration
            mock_posts = [
                {
                    "title": "Bitcoin adoption increasing in El Salvador",
                    "selftext": "Despite initial challenges, Bitcoin adoption in El Salvador is showing positive signs...",
                    "created_utc": time.time() - 3600,
                    "author": "bitcoin_enthusiast"
                },
                {
                    "title": "Ethereum 2.0 progress update",
                    "selftext": "The transition to Ethereum 2.0 is progressing well with the following milestones...",
                    "created_utc": time.time() - 7200,
                    "author": "eth_dev"
                },
                {
                    "title": "Analysis: Why ADA could be undervalued",
                    "selftext": "Looking at the fundamentals and upcoming developments, Cardano (ADA) might be undervalued...",
                    "created_utc": time.time() - 10800,
                    "author": "crypto_analyst"
                }
            ]
            
            result = []
            for post in mock_posts:
                # Combine title and text for better context
                full_text = f"{post.get('title', '')} {post.get('selftext', '')}"
                
                # Extract cryptocurrency symbols
                symbols = []
                for symbol in ["BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "XLM"]:
                    if symbol in full_text or symbol.lower() in full_text:
                        symbols.append(symbol)
                
                # Also check for full names
                crypto_names = {
                    "Bitcoin": "BTC",
                    "Ethereum": "ETH",
                    "Ripple": "XRP",
                    "Litecoin": "LTC",
                    "Cardano": "ADA",
                    "Polkadot": "DOT",
                    "Chainlink": "LINK",
                    "Binance": "BNB",
                    "Stellar": "XLM"
                }
                
                for name, symbol in crypto_names.items():
                    if name in full_text or name.lower() in full_text:
                        if symbol not in symbols:
                            symbols.append(symbol)
                
                # Create sentiment data item
                sentiment_item = SentimentData(
                    source=source.name,
                    text=post.get("selftext", ""),
                    title=post.get("title", ""),
                    published_at=datetime.fromtimestamp(post.get("created_utc", 0)),
                    author=post.get("author", ""),
                    symbols=symbols
                )
                
                result.append(sentiment_item)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _fetch_reddit_data: {str(e)}")
            return []
    
    async def _analyze_sentiment(self, data: SentimentData) -> Optional[SentimentResult]:
        """Analyze sentiment for a piece of text.
        
        Args:
            data: Sentiment data to analyze
            
        Returns:
            Sentiment analysis result or None if analysis fails
        """
        try:
            # Combine title and text for better context
            text = data.text
            if data.title:
                text = f"{data.title}. {text}"
            
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Determine whether to use the NLP model or simplified approach
            if not self.use_simplified_sentiment and self.sentiment_pipeline:
                # Use the NLP model
                result = self.sentiment_pipeline(text)[0]
                
                # Extract sentiment score and label
                label = result["label"]
                score = result["score"]
                
                # Convert label to standard format
                if label == "POS" or label == "positive":
                    sentiment_label = "positive"
                    sentiment_score = score
                elif label == "NEG" or label == "negative":
                    sentiment_label = "negative"
                    sentiment_score = -score
                else:
                    sentiment_label = "neutral"
                    sentiment_score = 0.0
                
                confidence = score
            else:
                # Use simplified rule-based sentiment analysis
                sentiment_score, sentiment_label, confidence = self._simplified_sentiment_analysis(text)
            
            # Calculate relevance score based on presence of symbols
            relevance_score = 0.5
            if data.symbols:
                relevance_score = 0.8
            
            # Create sentiment result
            result = SentimentResult(
                data_id=f"{data.source}_{int(time.time() * 1000)}",
                source=data.source,
                text_snippet=text[:100] + "..." if len(text) > 100 else text,
                symbols=data.symbols,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                timestamp=datetime.now(),
                relevance_score=relevance_score
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return None
    
    def _simplified_sentiment_analysis(self, text: str) -> Tuple[float, str, float]:
        """Simple rule-based sentiment analysis for testing purposes.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_score, sentiment_label, confidence)
        """
        # Convert to lowercase for easier matching
        text = text.lower()
        
        # Define positive and negative word lists
        positive_words = [
            "bullish", "positive", "good", "great", "excellent", "gain", "profit", 
            "increase", "up", "rising", "growth", "opportunity", "success", "promising",
            "optimistic", "rally", "recover", "breakthrough", "innovation", "adoption",
            "support", "progress", "potential", "strong", "confidence", "advantage"
        ]
        
        negative_words = [
            "bearish", "negative", "bad", "poor", "terrible", "loss", "crash", 
            "decrease", "down", "falling", "decline", "risk", "failure", "concerning",
            "pessimistic", "dump", "collapse", "sell-off", "problem", "issue",
            "resistance", "setback", "weak", "uncertainty", "disadvantage", "fear"
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0, "neutral", 0.6  # Neutral with moderate confidence
        
        sentiment_score = (positive_count - negative_count) / total_count
        
        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment_label = "positive"
        elif sentiment_score < -0.2:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        # Calculate confidence based on word count
        confidence = min(0.9, 0.5 + (total_count / 20))  # Cap at 0.9
        
        return sentiment_score, sentiment_label, confidence
    
    async def _aggregate_sentiment(self):
        """Aggregate sentiment data by cryptocurrency symbol."""
        try:
            # Group sentiment results by symbol
            symbol_results = {}
            
            # Get results from the last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_results = [r for r in self.sentiment_results if r.timestamp > cutoff_time]
            
            # Process each result
            for result in recent_results:
                for symbol in result.symbols:
                    if symbol not in symbol_results:
                        symbol_results[symbol] = []
                    symbol_results[symbol].append(result)
            
            # Calculate aggregate sentiment for each symbol
            for symbol, results in symbol_results.items():
                # Calculate weighted sentiment score
                weighted_scores = []
                source_scores = {}
                
                for result in results:
                    # Weight by relevance and recency
                    time_weight = 1.0
                    hours_ago = (datetime.now() - result.timestamp).total_seconds() / 3600
                    if hours_ago > 0:
                        time_weight = max(0.5, 1.0 - (hours_ago / 24))  # Decay over 24 hours
                    
                    weight = result.relevance_score * time_weight * result.confidence
                    weighted_scores.append((result.sentiment_score, weight))
                    
                    # Track sentiment by source
                    if result.source not in source_scores:
                        source_scores[result.source] = []
                    source_scores[result.source].append(result.sentiment_score)
                
                # Calculate overall sentiment score
                if weighted_scores:
                    total_score = sum(score * weight for score, weight in weighted_scores)
                    total_weight = sum(weight for _, weight in weighted_scores)
                    sentiment_score = total_score / total_weight if total_weight > 0 else 0.0
                else:
                    sentiment_score = 0.0
                
                # Determine sentiment label
                if sentiment_score > 0.2:
                    sentiment_label = "positive"
                elif sentiment_score < -0.2:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                # Calculate sentiment by source
                source_sentiment = {}
                for source, scores in source_scores.items():
                    source_sentiment[source] = sum(scores) / len(scores) if scores else 0.0
                
                # Calculate trend (change over time)
                trend = 0.0
                if len(results) >= 2:
                    # Sort by timestamp
                    sorted_results = sorted(results, key=lambda r: r.timestamp)
                    # Split into two halves
                    mid_point = len(sorted_results) // 2
                    first_half = sorted_results[:mid_point]
                    second_half = sorted_results[mid_point:]
                    # Calculate average sentiment for each half
                    first_avg = sum(r.sentiment_score for r in first_half) / len(first_half) if first_half else 0.0
                    second_avg = sum(r.sentiment_score for r in second_half) / len(second_half) if second_half else 0.0
                    # Calculate trend
                    trend = second_avg - first_avg
                
                # Get historical sentiment (last 7 days)
                historical = []
                for days_ago in range(7):
                    day_start = datetime.now() - timedelta(days=days_ago+1)
                    day_end = datetime.now() - timedelta(days=days_ago)
                    day_results = [r for r in self.sentiment_results 
                                  if r.timestamp > day_start and r.timestamp <= day_end and symbol in r.symbols]
                    if day_results:
                        day_score = sum(r.sentiment_score for r in day_results) / len(day_results)
                        historical.append((day_start, day_score))
                
                # Create aggregate sentiment
                aggregate = AggregateSentiment(
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    sentiment_label=sentiment_label,
                    volume=len(results),
                    trend=trend,
                    sources=source_sentiment,
                    timestamp=datetime.now(),
                    historical=historical
                )
                
                # Store in aggregate sentiments
                self.aggregate_sentiments[symbol] = aggregate
            
            logger.info(f"Aggregated sentiment for {len(symbol_results)} symbols")
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment: {str(e)}")
    
    async def _publish_sentiment_updates(self):
        """Publish sentiment updates to WebSocket channel."""
        if not self.websocket_channel:
            return
        
        try:
            # Publish aggregate sentiment
            message = {
                "type": "sentiment_update",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "symbols": {symbol: sentiment.dict() for symbol, sentiment in self.aggregate_sentiments.items()}
                }
            }
            
            await MessagePublisher.publish(self.websocket_channel, message)
            
        except Exception as e:
            logger.error(f"Error publishing sentiment updates: {str(e)}")
    
    async def get_sentiment(self, symbol: str) -> Optional[AggregateSentiment]:
        """Get aggregate sentiment for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC")
            
        Returns:
            Aggregate sentiment or None if not available
        """
        return self.aggregate_sentiments.get(symbol.upper())
    
    async def get_all_sentiments(self) -> Dict[str, AggregateSentiment]:
        """Get aggregate sentiment for all cryptocurrencies.
        
        Returns:
            Dictionary of aggregate sentiments by symbol
        """
        return self.aggregate_sentiments
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment.
        
        Returns:
            Market sentiment metrics
        """
        try:
            # Calculate overall market sentiment
            sentiments = list(self.aggregate_sentiments.values())
            
            if not sentiments:
                return {
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "fud_level": 0.0,
                    "fomo_level": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calculate weighted average sentiment
            total_score = 0.0
            total_volume = 0
            
            for sentiment in sentiments:
                total_score += sentiment.sentiment_score * sentiment.volume
                total_volume += sentiment.volume
            
            market_score = total_score / total_volume if total_volume > 0 else 0.0
            
            # Determine sentiment label
            if market_score > 0.2:
                market_label = "positive"
            elif market_score < -0.2:
                market_label = "negative"
            else:
                market_label = "neutral"
            
            # Calculate FUD level (Fear, Uncertainty, Doubt)
            negative_sentiments = [s for s in sentiments if s.sentiment_score < -0.3]
            fud_level = sum(abs(s.sentiment_score) * s.volume for s in negative_sentiments) / total_volume if total_volume > 0 else 0.0
            fud_level = min(1.0, fud_level * 2)  # Scale to 0-1
            
            # Calculate FOMO level (Fear Of Missing Out)
            positive_sentiments = [s for s in sentiments if s.sentiment_score > 0.3]
            fomo_level = sum(s.sentiment_score * s.volume for s in positive_sentiments) / total_volume if total_volume > 0 else 0.0
            fomo_level = min(1.0, fomo_level * 2)  # Scale to 0-1
            
            return {
                "sentiment_score": market_score,
                "sentiment_label": market_label,
                "fud_level": fud_level,
                "fomo_level": fomo_level,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_trading_signals(self) -> Dict[str, Any]:
        """Generate trading signals based on sentiment analysis.
        
        Returns:
            Dictionary of trading signals by symbol
        """
        try:
            signals = {}
            
            # Get market sentiment
            market_sentiment = await self.get_market_sentiment()
            
            # Check if FUD level is too high
            high_fud = market_sentiment.get("fud_level", 0.0) > 0.7
            
            # Process each cryptocurrency
            for symbol, sentiment in self.aggregate_sentiments.items():
                # Skip if volume is too low
                if sentiment.volume < 5:
                    continue
                
                # Determine signal
                signal = "neutral"
                strength = 0.0
                
                # If FUD level is high, avoid buy signals
                if high_fud and sentiment.sentiment_score > 0:
                    signal = "neutral"
                    reason = "High market FUD level, avoiding buy signals"
                else:
                    # Generate signal based on sentiment and trend
                    if sentiment.sentiment_score > 0.5 and sentiment.trend > 0.1:
                        signal = "strong_buy"
                        strength = min(1.0, sentiment.sentiment_score * 2)
                        reason = "Strong positive sentiment with improving trend"
                    elif sentiment.sentiment_score > 0.3:
                        signal = "buy"
                        strength = sentiment.sentiment_score
                        reason = "Positive sentiment"
                    elif sentiment.sentiment_score < -0.5 and sentiment.trend < -0.1:
                        signal = "strong_sell"
                        strength = min(1.0, abs(sentiment.sentiment_score) * 2)
                        reason = "Strong negative sentiment with worsening trend"
                    elif sentiment.sentiment_score < -0.3:
                        signal = "sell"
                        strength = abs(sentiment.sentiment_score)
                        reason = "Negative sentiment"
                    else:
                        signal = "neutral"
                        strength = 0.0
                        reason = "Neutral sentiment"
                
                # Create signal
                signals[symbol] = {
                    "symbol": symbol,
                    "signal": signal,
                    "strength": strength,
                    "reason": reason,
                    "sentiment_score": sentiment.sentiment_score,
                    "sentiment_trend": sentiment.trend,
                    "volume": sentiment.volume,
                    "timestamp": datetime.now().isoformat()
                }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return {"error": str(e)}
