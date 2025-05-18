"""
WebSocket Agent Module

This module provides WebSocket-enabled agents for real-time data streaming and
inter-agent communication. These agents use WebSockets instead of Kafka for
low-latency, bidirectional communication.

Key components:
- WebSocketAgent: Base agent that uses WebSockets for communication
- MarketDataAgent: Processes and publishes market data
- StrategyAgent: Executes trading strategies and publishes signals

This implementation allows for real-time data streaming between agents and with
frontend clients without the complexity of Kafka, providing a more lightweight
and browser-compatible solution.

Example usage:
    # Create and run a market data agent
    market_data_agent = MarketDataAgent()
    await market_data_agent.run({"symbol": "BTC-USDT", "price": 50000})
    
    # Create and run a strategy agent
    strategy_agent = StrategyAgent(strategy_id="sma-crossover")
    result = await strategy_agent.run(market_data)
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List
from app.agents.base_agent import BaseAgent, AgentAction, AgentObservation
from app.services.websocket_service import MessagePublisher

logger = logging.getLogger(__name__)


class WebSocketAgent(BaseAgent):
    """
    An agent that uses WebSockets for communication instead of Kafka.
    This agent can publish and subscribe to WebSocket channels for real-time data exchange.
    """
    
    def __init__(self, name: str, channels: List[str] = None, **kwargs):
        """
        Initialize the WebSocket agent.
        
        Args:
            name: Name of the agent
            channels: List of channels this agent will publish to
            **kwargs: Additional arguments to pass to BaseAgent
        """
        super().__init__(name, **kwargs)
        self.channels = channels or []
        
        # Register WebSocket-related tools
        self.register_tool(
            name="publish_to_channel",
            func=self.publish_to_channel,
            description="Publish data to a WebSocket channel"
        )
    
    async def publish_to_channel(self, channel: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish data to a WebSocket channel.
        
        Args:
            channel: Channel to publish to
            data: Data to publish
            
        Returns:
            Status of the publish operation
        """
        try:
            # Add metadata to the message
            message = {
                "sender": self.name,
                "timestamp": time.time(),
                "data": data
            }
            
            # Publish to the channel
            await MessagePublisher.publish(channel, message)
            
            return {"status": "success", "channel": channel}
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data and optionally publish results to WebSocket channels.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent's response
        """
        try:
            # Process the input data (implement your agent logic here)
            result = await self._process_data(input_data)
            
            # Publish results to configured channels if needed
            if hasattr(self, 'channels') and self.channels:
                for channel in self.channels:
                    await self.publish_to_channel(channel, result)
            
            return result
        except Exception as e:
            logger.error(f"Error in agent {self.name}: {str(e)}")
            return {"error": str(e)}
    
    async def _process_data(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data. Override this method in subclasses to implement
        specific agent logic.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Processed data
        """
        # Default implementation just returns the input data
        # Override this in subclasses to implement specific agent logic
        if isinstance(input_data, dict):
            return input_data
        else:
            return {"data": input_data}


class MarketDataAgent(WebSocketAgent):
    """
    Agent that processes market data and publishes it to WebSocket channels.
    """
    
    def __init__(self, name: str = "MarketDataAgent", **kwargs):
        """Initialize the MarketDataAgent."""
        super().__init__(name, channels=["market_data"], **kwargs)
    
    async def _process_data(self, input_data: Any) -> Dict[str, Any]:
        """
        Process market data.
        
        Args:
            input_data: Market data to process
            
        Returns:
            Processed market data
        """
        # Add any market data processing logic here
        # For example, calculate indicators, filter data, etc.
        
        if isinstance(input_data, dict):
            # Add a timestamp if not present
            if "timestamp" not in input_data:
                input_data["timestamp"] = time.time()
            
            # Add any derived data or indicators
            if "price" in input_data and "symbol" in input_data:
                # Example: Add a simple moving average if we have historical data
                # This is just a placeholder - in a real implementation, you would
                # maintain historical data and calculate actual indicators
                input_data["indicators"] = {
                    "simple_moving_average": input_data.get("price", 0)
                }
            
            return input_data
        else:
            return {"data": input_data, "timestamp": time.time()}


class StrategyAgent(WebSocketAgent):
    """
    Agent that executes trading strategies and publishes signals to WebSocket channels.
    """
    
    def __init__(self, name: str = "StrategyAgent", strategy_id: str = None, **kwargs):
        """
        Initialize the StrategyAgent.
        
        Args:
            name: Name of the agent
            strategy_id: ID of the strategy this agent is executing
            **kwargs: Additional arguments to pass to WebSocketAgent
        """
        super().__init__(name, channels=["strategy_signals", f"strategy_{strategy_id}"], **kwargs)
        self.strategy_id = strategy_id
    
    async def _process_data(self, input_data: Any) -> Dict[str, Any]:
        """
        Process market data and generate trading signals based on strategy.
        
        Args:
            input_data: Market data to process
            
        Returns:
            Trading signals
        """
        # Implement strategy logic here
        # For example, check conditions, generate buy/sell signals, etc.
        
        if not isinstance(input_data, dict):
            return {"error": "Invalid input data format"}
        
        # Example simple strategy logic (placeholder)
        signal = None
        if "price" in input_data and "symbol" in input_data:
            price = input_data.get("price", 0)
            
            # Very simple example logic - in a real implementation,
            # this would be based on your actual strategy rules
            if price > 0:
                # Example condition - replace with actual strategy logic
                if price % 2 == 0:  # Just a dummy condition
                    signal = "BUY"
                else:
                    signal = "SELL"
        
        return {
            "strategy_id": self.strategy_id,
            "timestamp": time.time(),
            "symbol": input_data.get("symbol"),
            "price": input_data.get("price"),
            "signal": signal,
            "metadata": {
                "source": self.name,
                "confidence": 0.8  # Example confidence score
            }
        }
