"""
Data Analyst Agent Module

This module implements a specialized agent for analyzing market data and generating
insights using the Autogen framework. The DataAnalystAgent can process market data,
identify patterns, and generate reports that can be used by other agents in the system.

Key features:
- Market data analysis with technical indicators
- Pattern recognition in price movements
- Sentiment analysis from news and social media
- Correlation analysis between different assets
- Anomaly detection in market behavior

This agent serves as the data intelligence component of the multi-agent system,
providing insights that drive strategy optimization and execution decisions.
"""

import logging
import asyncio
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from app.agents.autogen_agents import AutogenAgentManager, AutogenAgentConfig
from app.services.websocket_service import MessagePublisher
from app.core.okx_client import okx_client
from app.core.config import get_settings
from app.utils.data_utils import calculate_indicators

logger = logging.getLogger(__name__)
settings = get_settings()


class DataAnalystAgent:
    """Agent for analyzing market data and generating insights."""
    
    def __init__(self, websocket_channel: Optional[str] = None):
        """Initialize the data analyst agent.
        
        Args:
            websocket_channel: Optional channel for publishing analysis results
        """
        self.websocket_channel = websocket_channel
        self.autogen_manager = AutogenAgentManager()
        
        # Create the data analyst assistant agent
        self._create_agents()
        
        # Register functions with the user proxy agent
        self._register_functions()
    
    def _create_agents(self):
        """Create the Autogen agents for data analysis."""
        # Create the data analyst assistant
        data_analyst_config = AutogenAgentConfig(
            name="DataAnalystAssistant",
            system_prompt="""You are a Data Analyst Assistant specialized in cryptocurrency market analysis.
Your role is to analyze market data, identify patterns, and generate insights that can be used for trading.
You have expertise in technical analysis, pattern recognition, and statistical analysis.

Your responsibilities include:
1. Analyzing price and volume data to identify trends and patterns
2. Calculating and interpreting technical indicators
3. Identifying potential entry and exit points for trades
4. Detecting market anomalies and unusual behavior
5. Generating reports on market conditions and opportunities

Always base your analysis on data and avoid making speculative claims without evidence.
Provide clear, actionable insights that can be used by trading strategy agents.
""",
            websocket_channel=self.websocket_channel
        )
        
        # Create the user proxy agent for executing functions
        user_proxy_config = AutogenAgentConfig(
            name="DataAnalystProxy",
            system_prompt="""You are a Data Analyst Proxy that can execute functions to fetch and analyze market data.
Your role is to assist the Data Analyst Assistant by providing data and executing analysis functions.
You should always provide factual information based on the data you retrieve.
""",
            websocket_channel=self.websocket_channel
        )
        
        # Create the agents
        self.analyst_assistant = self.autogen_manager.create_agent(data_analyst_config)
        self.analyst_proxy = self.autogen_manager.create_agent(user_proxy_config)
    
    def _register_functions(self):
        """Register functions with the user proxy agent."""
        # Register data fetching functions
        self.autogen_manager.register_function(
            "DataAnalystProxy",
            self.fetch_market_data
        )
        
        # Register analysis functions
        self.autogen_manager.register_function(
            "DataAnalystProxy",
            self.analyze_price_action
        )
        
        self.autogen_manager.register_function(
            "DataAnalystProxy",
            self.calculate_technical_indicators
        )
        
        self.autogen_manager.register_function(
            "DataAnalystProxy",
            self.detect_patterns
        )
        
        self.autogen_manager.register_function(
            "DataAnalystProxy",
            self.publish_analysis
        )
    
    async def analyze_market(self, 
                           trading_pair: str, 
                           timeframe: str = "1h", 
                           lookback_periods: int = 100) -> Dict[str, Any]:
        """Analyze market data for a trading pair.
        
        Args:
            trading_pair: Trading pair to analyze (e.g., "BTC-USDT")
            timeframe: Timeframe for the analysis (e.g., "1m", "5m", "1h", "1d")
            lookback_periods: Number of periods to look back
            
        Returns:
            Analysis results
        """
        # Start a conversation between the agents
        conversation_id = await self.autogen_manager.start_conversation(
            "DataAnalystProxy",
            "DataAnalystAssistant",
            f"""Please analyze the market data for {trading_pair} on the {timeframe} timeframe.
Fetch the last {lookback_periods} periods of data and perform the following analysis:
1. Calculate key technical indicators
2. Identify any significant patterns
3. Determine the current market trend
4. Identify potential support and resistance levels
5. Provide a summary of market conditions

Please be thorough in your analysis and provide actionable insights.
"""
        )
        
        # Get the conversation results
        conversation = self.autogen_manager.conversations[conversation_id]
        
        # Extract the analysis from the conversation
        analysis_message = conversation["messages"][-1]["content"]
        
        # Create a structured analysis result
        analysis_result = {
            "trading_pair": trading_pair,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis_message,
            "conversation_id": conversation_id
        }
        
        # Publish the analysis to the websocket channel
        if self.websocket_channel:
            await MessagePublisher.publish(
                self.websocket_channel,
                {
                    "type": "market_analysis",
                    "data": analysis_result
                }
            )
        
        return analysis_result
    
    async def fetch_market_data(self, trading_pair: str, timeframe: str = "1h", limit: int = 100) -> Dict[str, Any]:
        """Fetch market data for a trading pair.
        
        Args:
            trading_pair: Trading pair to fetch data for
            timeframe: Timeframe for the data
            limit: Number of candles to fetch
            
        Returns:
            Market data
        """
        try:
            # Fetch data from OKX
            response = await okx_client.get_kline_data(trading_pair, bar=timeframe, limit=limit)
            
            # Check for errors
            if "error" in response:
                return {"error": response["error"]}
            
            # Extract the data
            candles = response.get("data", [])
            
            # Convert to a more usable format
            data = []
            for candle in candles:
                # OKX format: [timestamp, open, high, low, close, volume, ...]
                data.append({
                    "timestamp": candle[0],
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })
            
            return {
                "trading_pair": trading_pair,
                "timeframe": timeframe,
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_price_action(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price action from market data.
        
        Args:
            market_data: Market data from fetch_market_data
            
        Returns:
            Price action analysis
        """
        try:
            # Extract the data
            data = market_data.get("data", [])
            if not data:
                return {"error": "No data to analyze"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Calculate basic statistics
            latest_price = df["close"].iloc[-1]
            price_change = (latest_price / df["close"].iloc[0] - 1) * 100
            high_price = df["high"].max()
            low_price = df["low"].min()
            avg_volume = df["volume"].mean()
            
            # Determine trend
            sma20 = df["close"].rolling(20).mean().iloc[-1]
            sma50 = df["close"].rolling(50).mean().iloc[-1]
            
            if latest_price > sma20 and sma20 > sma50:
                trend = "bullish"
            elif latest_price < sma20 and sma20 < sma50:
                trend = "bearish"
            else:
                trend = "neutral"
            
            # Calculate volatility
            returns = df["close"].pct_change().dropna()
            volatility = returns.std() * 100
            
            # Identify potential support and resistance levels
            pivots = self._find_pivot_points(df)
            
            return {
                "latest_price": latest_price,
                "price_change_percent": price_change,
                "high_price": high_price,
                "low_price": low_price,
                "avg_volume": avg_volume,
                "trend": trend,
                "volatility_percent": volatility,
                "support_levels": pivots["support"],
                "resistance_levels": pivots["resistance"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price action: {str(e)}")
            return {"error": str(e)}
    
    async def calculate_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators from market data.
        
        Args:
            market_data: Market data from fetch_market_data
            
        Returns:
            Technical indicators
        """
        try:
            # Extract the data
            data = market_data.get("data", [])
            if not data:
                return {"error": "No data to analyze"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Calculate indicators
            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            
            # Bollinger Bands
            sma20 = df["close"].rolling(20).mean()
            std20 = df["close"].rolling(20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            
            # Stochastic Oscillator
            low_14 = df["low"].rolling(14).min()
            high_14 = df["high"].rolling(14).max()
            k = 100 * ((df["close"] - low_14) / (high_14 - low_14))
            d = k.rolling(3).mean()
            
            return {
                "rsi": rsi.iloc[-1],
                "macd": macd.iloc[-1],
                "macd_signal": signal.iloc[-1],
                "macd_histogram": (macd - signal).iloc[-1],
                "bollinger_middle": sma20.iloc[-1],
                "bollinger_upper": upper_band.iloc[-1],
                "bollinger_lower": lower_band.iloc[-1],
                "stochastic_k": k.iloc[-1],
                "stochastic_d": d.iloc[-1],
                "sma20": sma20.iloc[-1],
                "sma50": df["close"].rolling(50).mean().iloc[-1],
                "sma200": df["close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {"error": str(e)}
    
    async def detect_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns in market data.
        
        Args:
            market_data: Market data from fetch_market_data
            
        Returns:
            Detected patterns
        """
        try:
            # Extract the data
            data = market_data.get("data", [])
            if not data or len(data) < 5:
                return {"error": "Insufficient data to detect patterns"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Detect patterns
            patterns = []
            
            # Detect trend reversals
            if self._is_double_bottom(df):
                patterns.append("double_bottom")
            
            if self._is_double_top(df):
                patterns.append("double_top")
            
            # Detect candlestick patterns
            if self._is_doji(df.iloc[-1]):
                patterns.append("doji")
            
            if self._is_hammer(df.iloc[-1]):
                patterns.append("hammer")
            
            if self._is_engulfing(df.iloc[-2:]):
                patterns.append("engulfing")
            
            # Detect breakouts
            if self._is_breakout(df):
                patterns.append("breakout")
            
            return {
                "patterns": patterns,
                "pattern_descriptions": self._get_pattern_descriptions(patterns)
            }
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return {"error": str(e)}
    
    async def publish_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Publish analysis results to the websocket channel.
        
        Args:
            analysis: Analysis results to publish
            
        Returns:
            Status of the publish operation
        """
        try:
            if not self.websocket_channel:
                return {"status": "error", "message": "No websocket channel configured"}
            
            # Add timestamp
            analysis["timestamp"] = datetime.now().isoformat()
            
            # Publish to the websocket channel
            await MessagePublisher.publish(
                self.websocket_channel,
                {
                    "type": "market_analysis",
                    "data": analysis
                }
            )
            
            return {"status": "success", "message": "Analysis published successfully"}
            
        except Exception as e:
            logger.error(f"Error publishing analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _find_pivot_points(self, df: pd.DataFrame, window: int = 5) -> Dict[str, List[float]]:
        """Find pivot points (support and resistance levels) in price data.
        
        Args:
            df: DataFrame with price data
            window: Window size for pivot detection
            
        Returns:
            Dictionary with support and resistance levels
        """
        pivots = {"support": [], "resistance": []}
        
        # Find local minima (support)
        for i in range(window, len(df) - window):
            if all(df["low"].iloc[i] <= df["low"].iloc[i-j] for j in range(1, window+1)) and \
               all(df["low"].iloc[i] <= df["low"].iloc[i+j] for j in range(1, window+1)):
                pivots["support"].append(df["low"].iloc[i])
        
        # Find local maxima (resistance)
        for i in range(window, len(df) - window):
            if all(df["high"].iloc[i] >= df["high"].iloc[i-j] for j in range(1, window+1)) and \
               all(df["high"].iloc[i] >= df["high"].iloc[i+j] for j in range(1, window+1)):
                pivots["resistance"].append(df["high"].iloc[i])
        
        # Limit to the 5 most recent levels
        pivots["support"] = sorted(pivots["support"])[-5:]
        pivots["resistance"] = sorted(pivots["resistance"])[-5:]
        
        return pivots
    
    def _is_double_bottom(self, df: pd.DataFrame) -> bool:
        """Check if there's a double bottom pattern in the data."""
        # Simplified implementation
        if len(df) < 20:
            return False
        
        # Look for two similar lows with a higher point in between
        lows = df["low"].rolling(5).min()
        if len(lows.dropna()) < 15:
            return False
        
        # Find local minima
        local_mins = []
        for i in range(5, len(lows) - 5):
            if lows.iloc[i] == min(lows.iloc[i-5:i+6]):
                local_mins.append((i, lows.iloc[i]))
        
        # Need at least 2 minima
        if len(local_mins) < 2:
            return False
        
        # Check if the last two minima are similar in price
        if len(local_mins) >= 2:
            last_min = local_mins[-1]
            second_last_min = local_mins[-2]
            
            # Check if prices are within 2% of each other
            price_diff = abs(last_min[1] - second_last_min[1]) / second_last_min[1]
            if price_diff < 0.02:
                # Check if there's a peak in between
                between_idx = range(second_last_min[0], last_min[0])
                if len(between_idx) > 0:
                    between_high = max(df["high"].iloc[between_idx])
                    if between_high > last_min[1] * 1.03:  # At least 3% higher
                        return True
        
        return False
    
    def _is_double_top(self, df: pd.DataFrame) -> bool:
        """Check if there's a double top pattern in the data."""
        # Simplified implementation
        if len(df) < 20:
            return False
        
        # Look for two similar highs with a lower point in between
        highs = df["high"].rolling(5).max()
        if len(highs.dropna()) < 15:
            return False
        
        # Find local maxima
        local_maxs = []
        for i in range(5, len(highs) - 5):
            if highs.iloc[i] == max(highs.iloc[i-5:i+6]):
                local_maxs.append((i, highs.iloc[i]))
        
        # Need at least 2 maxima
        if len(local_maxs) < 2:
            return False
        
        # Check if the last two maxima are similar in price
        if len(local_maxs) >= 2:
            last_max = local_maxs[-1]
            second_last_max = local_maxs[-2]
            
            # Check if prices are within 2% of each other
            price_diff = abs(last_max[1] - second_last_max[1]) / second_last_max[1]
            if price_diff < 0.02:
                # Check if there's a trough in between
                between_idx = range(second_last_max[0], last_max[0])
                if len(between_idx) > 0:
                    between_low = min(df["low"].iloc[between_idx])
                    if between_low < last_max[1] * 0.97:  # At least 3% lower
                        return True
        
        return False
    
    def _is_doji(self, candle: pd.Series) -> bool:
        """Check if a candle is a doji (open and close are very close)."""
        body_size = abs(candle["close"] - candle["open"])
        candle_range = candle["high"] - candle["low"]
        
        # Body is less than 10% of the total range
        return body_size <= candle_range * 0.1
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """Check if a candle is a hammer (long lower wick, small body, little/no upper wick)."""
        body_size = abs(candle["close"] - candle["open"])
        candle_range = candle["high"] - candle["low"]
        
        if candle_range == 0:
            return False
        
        body_top = max(candle["open"], candle["close"])
        body_bottom = min(candle["open"], candle["close"])
        
        upper_wick = candle["high"] - body_top
        lower_wick = body_bottom - candle["low"]
        
        # Lower wick is at least 2x the body size
        # Upper wick is small
        return (lower_wick >= body_size * 2 and 
                upper_wick <= body_size * 0.5 and
                body_size <= candle_range * 0.3)
    
    def _is_engulfing(self, candles: pd.DataFrame) -> bool:
        """Check if the last two candles form an engulfing pattern."""
        if len(candles) < 2:
            return False
        
        prev_candle = candles.iloc[0]
        curr_candle = candles.iloc[1]
        
        prev_body_top = max(prev_candle["open"], prev_candle["close"])
        prev_body_bottom = min(prev_candle["open"], prev_candle["close"])
        curr_body_top = max(curr_candle["open"], curr_candle["close"])
        curr_body_bottom = min(curr_candle["open"], curr_candle["close"])
        
        # Bullish engulfing
        if (prev_candle["close"] < prev_candle["open"] and  # Previous is bearish
            curr_candle["close"] > curr_candle["open"] and  # Current is bullish
            curr_body_bottom <= prev_body_bottom and        # Current engulfs previous
            curr_body_top >= prev_body_top):
            return True
        
        # Bearish engulfing
        if (prev_candle["close"] > prev_candle["open"] and  # Previous is bullish
            curr_candle["close"] < curr_candle["open"] and  # Current is bearish
            curr_body_bottom <= prev_body_bottom and        # Current engulfs previous
            curr_body_top >= prev_body_top):
            return True
        
        return False
    
    def _is_breakout(self, df: pd.DataFrame) -> bool:
        """Check if there's a breakout in the data."""
        if len(df) < 20:
            return False
        
        # Calculate 20-day high and low
        high_20 = df["high"].rolling(20).max().shift(1)
        low_20 = df["low"].rolling(20).min().shift(1)
        
        # Check if the latest close breaks above the 20-day high
        # or below the 20-day low with increased volume
        latest = df.iloc[-1]
        avg_volume = df["volume"].iloc[-21:-1].mean()
        
        if latest["close"] > high_20.iloc[-1] and latest["volume"] > avg_volume * 1.5:
            return True  # Bullish breakout
        
        if latest["close"] < low_20.iloc[-1] and latest["volume"] > avg_volume * 1.5:
            return True  # Bearish breakout
        
        return False
    
    def _get_pattern_descriptions(self, patterns: List[str]) -> Dict[str, str]:
        """Get descriptions for detected patterns."""
        descriptions = {
            "double_bottom": "A bullish reversal pattern where price forms two lows at approximately the same level, indicating potential trend reversal.",
            "double_top": "A bearish reversal pattern where price forms two highs at approximately the same level, indicating potential trend reversal.",
            "doji": "A candlestick with a small body, indicating indecision in the market.",
            "hammer": "A bullish reversal pattern with a small body and long lower wick, often appearing at the bottom of a downtrend.",
            "engulfing": "A two-candle pattern where the second candle completely engulfs the body of the first, indicating potential reversal.",
            "breakout": "Price breaking above resistance or below support with increased volume, indicating potential trend continuation."
        }
        
        return {pattern: descriptions.get(pattern, "Unknown pattern") for pattern in patterns}


# Create a singleton instance
data_analyst_agent = DataAnalystAgent(websocket_channel="market_analysis")
