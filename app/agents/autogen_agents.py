"""
Autogen Agents Module

This module implements autonomous agents using the Autogen framework for data analysis
and strategy execution in the AI Strategy Lab. These agents can work together to analyze
market data, optimize strategies, and execute trades with minimal human intervention.

Key components:
- DataAnalystAgent: Analyzes market data and generates insights
- StrategyOptimizerAgent: Optimizes trading strategies based on data insights
- ExecutionAgent: Executes trades based on strategy signals
- AgentCoordinator: Coordinates communication between agents

This implementation allows for autonomous operation of the trading system with agents
that can communicate with each other and make decisions based on real-time data.
"""

import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable, Union
from pydantic import BaseModel, Field
import autogen
from autogen.agentchat.agent import Agent
from autogen.agentchat.assistant import AssistantAgent
from autogen.agentchat.user_proxy import UserProxyAgent

from app.agents.base_agent import BaseAgent
from app.services.websocket_service import MessagePublisher
from app.services.rl_service import rl_service
from app.core.okx_client import okx_client
from app.core.config import get_settings
from app.models.strategy import Strategy, TradeSignal

logger = logging.getLogger(__name__)
settings = get_settings()


class AutogenAgentConfig(BaseModel):
    """Configuration for an Autogen agent."""
    name: str
    system_prompt: str
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    human_input_mode: str = "NEVER"
    max_consecutive_auto_reply: int = 10
    websocket_channel: Optional[str] = None


class AutogenAgentManager:
    """Manager for Autogen agents in the AI Strategy Lab."""
    
    def __init__(self):
        """Initialize the Autogen agent manager."""
        self.agents = {}
        self.conversations = {}
        
        # Configure default LLM settings
        self.default_llm_config = {
            "config_list": [
                {
                    "model": settings.LLM_MODEL,
                    "api_key": settings.OPENAI_API_KEY,
                }
            ],
            "temperature": 0.1,
            "timeout": 120,
        }
    
    def create_agent(self, config: AutogenAgentConfig) -> Agent:
        """Create an Autogen agent with the specified configuration.
        
        Args:
            config: Configuration for the agent
            
        Returns:
            The created agent
        """
        # Use provided LLM config or default
        llm_config = config.llm_config or self.default_llm_config
        
        # Create the agent
        if "assistant" in config.name.lower():
            agent = AssistantAgent(
                name=config.name,
                system_message=config.system_prompt,
                llm_config=llm_config,
                human_input_mode=config.human_input_mode,
                max_consecutive_auto_reply=config.max_consecutive_auto_reply,
            )
        else:
            # Create a UserProxyAgent for agents that need to execute functions
            agent = UserProxyAgent(
                name=config.name,
                system_message=config.system_prompt,
                human_input_mode=config.human_input_mode,
                max_consecutive_auto_reply=config.max_consecutive_auto_reply,
                # We'll register functions later
            )
        
        # Store the agent
        self.agents[config.name] = agent
        
        # Store websocket channel for progress updates
        if config.websocket_channel:
            setattr(agent, "websocket_channel", config.websocket_channel)
        
        return agent
    
    def register_function(self, agent_name: str, function: Callable, function_name: Optional[str] = None):
        """Register a function with a UserProxyAgent.
        
        Args:
            agent_name: Name of the agent to register the function with
            function: Function to register
            function_name: Name of the function (defaults to function.__name__)
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        if not isinstance(agent, UserProxyAgent):
            raise ValueError(f"Agent {agent_name} is not a UserProxyAgent")
        
        # Register the function
        function_name = function_name or function.__name__
        agent.register_function(function, name=function_name)
    
    async def start_conversation(self, 
                                initiator_name: str, 
                                responder_name: str, 
                                message: str,
                                conversation_id: Optional[str] = None) -> str:
        """Start a conversation between two agents.
        
        Args:
            initiator_name: Name of the agent initiating the conversation
            responder_name: Name of the agent responding to the initiator
            message: Initial message to send
            conversation_id: Optional ID for the conversation
            
        Returns:
            Conversation ID
        """
        if initiator_name not in self.agents:
            raise ValueError(f"Initiator agent {initiator_name} not found")
        if responder_name not in self.agents:
            raise ValueError(f"Responder agent {responder_name} not found")
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"{initiator_name}_{responder_name}_{int(time.time())}"
        
        # Get the agents
        initiator = self.agents[initiator_name]
        responder = self.agents[responder_name]
        
        # Start the conversation
        chat_result = await initiator.a_initiate_chat(
            responder,
            message=message,
            clear_history=True,
        )
        
        # Store the conversation
        self.conversations[conversation_id] = {
            "initiator": initiator_name,
            "responder": responder_name,
            "messages": [
                {"role": "initiator", "content": message},
                {"role": "responder", "content": str(chat_result)},
            ],
            "timestamp": time.time(),
        }
        
        # Publish conversation to websocket if channel is configured
        if hasattr(initiator, "websocket_channel") and initiator.websocket_channel:
            await self._publish_conversation(conversation_id, initiator.websocket_channel)
        
        return conversation_id
    
    async def continue_conversation(self, conversation_id: str, message: str) -> str:
        """Continue an existing conversation between agents.
        
        Args:
            conversation_id: ID of the conversation to continue
            message: Message to send
            
        Returns:
            Response from the responder agent
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Get the conversation
        conversation = self.conversations[conversation_id]
        
        # Get the agents
        initiator = self.agents[conversation["initiator"]]
        responder = self.agents[conversation["responder"]]
        
        # Continue the conversation
        chat_result = await initiator.a_send(
            message=message,
            recipient=responder,
        )
        
        # Update the conversation
        conversation["messages"].extend([
            {"role": "initiator", "content": message},
            {"role": "responder", "content": str(chat_result)},
        ])
        conversation["timestamp"] = time.time()
        
        # Publish conversation to websocket if channel is configured
        if hasattr(initiator, "websocket_channel") and initiator.websocket_channel:
            await self._publish_conversation(conversation_id, initiator.websocket_channel)
        
        return str(chat_result)
    
    async def _publish_conversation(self, conversation_id: str, channel: str):
        """Publish a conversation to a websocket channel.
        
        Args:
            conversation_id: ID of the conversation to publish
            channel: Websocket channel to publish to
        """
        if conversation_id not in self.conversations:
            return
        
        # Get the conversation
        conversation = self.conversations[conversation_id]
        
        # Create message
        message = {
            "type": "agent_conversation",
            "conversation_id": conversation_id,
            "initiator": conversation["initiator"],
            "responder": conversation["responder"],
            "messages": conversation["messages"],
            "timestamp": conversation["timestamp"],
        }
        
        # Publish to websocket
        await MessagePublisher.publish(channel, message)


# Create a singleton instance
autogen_manager = AutogenAgentManager()
