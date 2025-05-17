import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel
from langchain_core.language_models import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentAction(BaseModel):
    """Action that an agent can take."""
    name: str
    args: Dict[str, Any] = {}
    description: Optional[str] = None


class AgentObservation(BaseModel):
    """Observation from the environment."""
    data: Any
    source: str
    timestamp: float


class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, llm: Optional[BaseLLM] = None, chat_model: Optional[BaseChatModel] = None):
        """Initialize the agent.
        
        Args:
            name: Name of the agent
            llm: Language model for reasoning
            chat_model: Chat model for conversation
        """
        self.name = name
        self.llm = llm
        self.chat_model = chat_model
        self.memory = []  # Store conversation history
        self.tools = {}  # Available tools/functions
        self.system_prompt = f"You are {name}, an AI agent specialized in crypto trading strategies."
    
    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool that the agent can use.
        
        Args:
            name: Name of the tool
            func: Function to call when tool is used
            description: Description of what the tool does
        """
        self.tools[name] = {
            "func": func,
            "description": description
        }
    
    async def run(self, input_data: Any) -> Dict[str, Any]:
        """Run the agent on input data.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent's response
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the language model.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            LLM response
        """
        if not self.llm:
            raise ValueError(f"Agent {self.name} has no LLM configured")
        
        try:
            return await self.llm.agenerate([prompt])
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _call_chat_model(self, messages: List[Dict[str, str]]) -> str:
        """Call the chat model.
        
        Args:
            messages: List of messages for the chat model
            
        Returns:
            Chat model response
        """
        if not self.chat_model:
            raise ValueError(f"Agent {self.name} has no chat model configured")
        
        try:
            # Convert dict messages to LangChain message objects
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            response = await self.chat_model.agenerate([lc_messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"Error calling chat model: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _execute_tool(self, action: AgentAction) -> Any:
        """Execute a tool.
        
        Args:
            action: Action to execute
            
        Returns:
            Tool execution result
        """
        if action.name not in self.tools:
            raise ValueError(f"Tool {action.name} not found")
        
        try:
            tool = self.tools[action.name]
            result = await tool["func"](**action.args)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {action.name}: {str(e)}")
            return {"error": str(e)}
    
    def _add_to_memory(self, role: str, content: str):
        """Add a message to agent memory.
        
        Args:
            role: Role of the message sender (system, user, assistant)
            content: Message content
        """
        self.memory.append({"role": role, "content": content})
    
    def _get_memory(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages from memory.
        
        Args:
            last_n: Number of most recent messages to retrieve
            
        Returns:
            List of messages
        """
        if last_n is None:
            return self.memory
        return self.memory[-last_n:] if last_n < len(self.memory) else self.memory
    
    def _clear_memory(self):
        """Clear agent memory."""
        self.memory = []
    
    def __str__(self) -> str:
        return f"Agent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
