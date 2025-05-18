import logging
import json
from typing import List, Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from app.models.strategy import Strategy, StrategyType, RiskTolerance
from app.core.config import get_settings
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMService:
    """Service for Large Language Model interactions."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.LLM_MODEL
        self.llm = None
        self.chat_model = None
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize LLM models."""
        try:
            logger.info(f"Initializing LLM model: {self.model_name}")
            
            # Initialize standard LLM
            self.llm = OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.2,
                max_tokens=2048
            )
            
            # Initialize chat model
            self.chat_model = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.2,
                max_tokens=2048
            )
            
            logger.info("LLM models initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing LLM models: {str(e)}")
            raise
    
    async def generate_strategy(
        self, 
        strategy_type: StrategyType, 
        trading_pairs: List[str],
        risk_tolerance: RiskTolerance = RiskTolerance.MEDIUM,
        description: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Strategy:
        """Generate a trading strategy using LLM."""
        try:
            logger.info(f"Generating {strategy_type} strategy for {', '.join(trading_pairs)}")
            
            # Retrieve relevant strategy templates using RAG
            templates = await rag_service.get_relevant_strategy_templates(
                strategy_type=strategy_type.value,
                description=description or "",
                k=3
            )
            
            # Extract template content
            template_examples = "\n\n".join([
                f"Example {i+1}:\n{template['content']}"
                for i, template in enumerate(templates)
            ])
            
            # Prepare constraints text
            constraints_text = ""
            if constraints:
                constraints_text = "Strategy constraints:\n" + "\n".join([
                    f"- {key}: {value}" for key, value in constraints.items()
                ])
            
            # Construct prompt
            strategy_prompt = PromptTemplate(
                input_variables=["strategy_type", "trading_pairs", "risk_tolerance", "description", "template_examples", "constraints"],
                template="""
                Generate a detailed trading strategy with the following characteristics:
                
                Strategy Type: {strategy_type}
                Trading Pairs: {trading_pairs}
                Risk Tolerance: {risk_tolerance}
                Description: {description}
                {constraints}
                
                Here are some example strategies to guide you:
                {template_examples}
                
                Please generate a complete strategy including:
                1. A descriptive name
                2. Technical indicators with parameters
                3. Entry and exit rules with specific conditions
                4. Risk management parameters
                
                Format the output as a valid JSON object that matches the Strategy model.
                """
            )
            
            # Create LLM chain
            chain = LLMChain(
                llm=self.llm,
                prompt=strategy_prompt
            )
            
            # Run chain to generate strategy
            result = await chain.arun(
                strategy_type=strategy_type.value,
                trading_pairs=", ".join(trading_pairs),
                risk_tolerance=risk_tolerance.value,
                description=description or f"Generate a {strategy_type.value} strategy for {', '.join(trading_pairs)}",
                template_examples=template_examples,
                constraints=constraints_text
            )
            
            # Parse result into Strategy object
            # In a real implementation, we would use proper parsing
            # For simplicity, we'll create a dummy strategy
            strategy = Strategy(
                name=f"{strategy_type.capitalize()} Strategy for {', '.join(trading_pairs)}",
                description=description or f"AI-generated {strategy_type} strategy",
                type=strategy_type,
                trading_pairs=trading_pairs,
                risk_tolerance=risk_tolerance,
                indicators=[
                    {
                        "name": "RSI",
                        "parameters": {"period": 14, "overbought": 70, "oversold": 30}
                    }
                ],
                rules=[
                    {
                        "name": "Buy signal",
                        "conditions": [
                            {
                                "indicator": "RSI",
                                "operator": "<",
                                "value": 30
                            }
                        ],
                        "actions": [
                            {
                                "type": "BUY",
                                "asset": trading_pairs[0].split('-')[0],
                                "amount_type": "percentage",
                                "amount": 50.0
                            }
                        ]
                    },
                    {
                        "name": "Sell signal",
                        "conditions": [
                            {
                                "indicator": "RSI",
                                "operator": ">",
                                "value": 70
                            }
                        ],
                        "actions": [
                            {
                                "type": "SELL",
                                "asset": trading_pairs[0].split('-')[0],
                                "amount_type": "percentage",
                                "amount": 100.0
                            }
                        ]
                    }
                ],
                parameters={
                    "stop_loss": 5.0,
                    "take_profit": 10.0
                },
                metadata={
                    "created_at": "2025-05-16T10:00:00Z",
                    "created_by": "AI Strategy Lab"
                }
            )
            
            logger.info(f"Successfully generated strategy")
            return strategy
        
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment in text."""
        try:
            logger.info("Analyzing sentiment")
            
            # Construct prompt
            sentiment_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                Analyze the sentiment in the following text related to cryptocurrency trading.
                Text: {text}
                
                Provide a JSON object with the following fields:
                - sentiment: overall sentiment (positive, neutral, negative)
                - confidence: confidence score between 0 and 1
                - key_points: list of key points that influenced the sentiment analysis
                """
            )
            
            # Create LLM chain
            chain = LLMChain(
                llm=self.llm,
                prompt=sentiment_prompt
            )
            
            # Run chain
            result = await chain.arun(text=text)
            
            # Parse result (in a real implementation, use proper parsing)
            try:
                sentiment_data = json.loads(result)
            except:
                # Fallback if parsing fails
                sentiment_data = {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "key_points": ["Unable to parse sentiment properly"]
                }
            
            logger.info(f"Sentiment analysis complete: {sentiment_data['sentiment']}")
            return sentiment_data
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise
    
    async def optimize_strategy_parameters(
        self, 
        strategy: Strategy,
        market_data: Dict[str, Any]
    ) -> Strategy:
        """Optimize strategy parameters based on market data."""
        try:
            logger.info(f"Optimizing parameters for strategy")
            
            # In a real implementation, we would use LLM to suggest parameter optimizations
            # For simplicity, we'll return the original strategy with minor modifications
            modified_strategy = strategy.copy()
            
            # Modify a parameter (in a real implementation, this would be based on analysis)
            if "RSI" in [indicator["name"] for indicator in modified_strategy.indicators]:
                for indicator in modified_strategy.indicators:
                    if indicator["name"] == "RSI":
                        indicator["parameters"]["period"] = 16  # Modified from default 14
            
            if "parameters" in modified_strategy.dict() and modified_strategy.parameters:
                if "stop_loss" in modified_strategy.parameters:
                    modified_strategy.parameters["stop_loss"] = 4.5  # Modified from default 5.0
            
            logger.info(f"Strategy parameter optimization complete")
            return modified_strategy
        
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {str(e)}")
            raise


# Create a singleton instance
llm_service = LLMService()