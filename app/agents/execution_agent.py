import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from app.agents.base_agent import BaseAgent, AgentAction, AgentObservation
from app.models.strategy import Strategy, StrategyPerformance
from app.services.execution_service import execution_service
from app.core.okx_client import okx_client
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ExecutionAgent(BaseAgent):
    """Agent responsible for executing trading strategies via OKX API."""
    
    def __init__(self):
        """Initialize the execution agent."""
        super().__init__(name="Execution Agent")
        
        # Initialize LLM if API key is available
        if settings.OPENAI_API_KEY:
            self.chat_model = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.LLM_MODEL,
                temperature=0.1  # Lower temperature for more conservative decisions
            )
        
        # Register tools
        self.register_tool(
            name="deploy_strategy",
            func=self._deploy_strategy,
            description="Deploy a strategy for live trading"
        )
        self.register_tool(
            name="stop_strategy",
            func=self._stop_strategy,
            description="Stop a deployed strategy"
        )
        self.register_tool(
            name="get_strategy_performance",
            func=self._get_strategy_performance,
            description="Get performance metrics for a deployed strategy"
        )
        self.register_tool(
            name="execute_manual_trade",
            func=self._execute_manual_trade,
            description="Execute a manual trade for a specific trading pair"
        )
        self.register_tool(
            name="check_market_conditions",
            func=self._check_market_conditions,
            description="Check current market conditions for a trading pair"
        )
        
        # System prompt
        self.system_prompt = """
        You are the Execution Agent, an AI specialized in executing trading strategies on cryptocurrency exchanges.
        Your primary responsibilities are:
        1. Deploy strategies for live trading
        2. Monitor strategy performance
        3. Execute trades based on strategy signals
        4. Ensure proper risk management during execution
        5. Provide real-time feedback on strategy performance
        
        You have expertise in order execution, market microstructure, and risk management.
        Always prioritize capital preservation and risk management over potential profits.
        """
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on input data.
        
        Args:
            input_data: Input data containing action and parameters
            
        Returns:
            Execution results
        """
        try:
            logger.info("Running Execution Agent")
            
            # Extract input data
            action = input_data.get("action")
            
            # Validate input
            if not action:
                return {"error": "Action is required"}
            
            # Determine action
            if action == "deploy":
                strategy = input_data.get("strategy")
                if not strategy:
                    return {"error": "Strategy is required for deployment"}
                
                # Check if we should validate market conditions before deployment
                validate_market = input_data.get("validate_market", True)
                if validate_market:
                    # Check market conditions for all trading pairs
                    for trading_pair in strategy.trading_pairs:
                        market_conditions = await self._check_market_conditions(trading_pair)
                        if market_conditions.get("status") == "unfavorable":
                            return {
                                "status": "rejected",
                                "reason": f"Unfavorable market conditions for {trading_pair}: {market_conditions.get('details')}"
                            }
                
                # Deploy the strategy
                deployment_result = await self._deploy_strategy(strategy)
                return deployment_result
                
            elif action == "stop":
                strategy_id = input_data.get("strategy_id")
                if not strategy_id:
                    return {"error": "Strategy ID is required to stop a strategy"}
                
                # Stop the strategy
                stop_result = await self._stop_strategy(strategy_id)
                return stop_result
                
            elif action == "performance":
                strategy_id = input_data.get("strategy_id")
                if not strategy_id:
                    return {"error": "Strategy ID is required to get performance"}
                
                # Get performance
                performance = await self._get_strategy_performance(strategy_id)
                return {"performance": performance}
                
            elif action == "trade":
                trading_pair = input_data.get("trading_pair")
                side = input_data.get("side")
                amount = input_data.get("amount")
                price = input_data.get("price")
                
                if not all([trading_pair, side, amount]):
                    return {"error": "Trading pair, side, and amount are required for manual trade"}
                
                # Execute manual trade
                trade_result = await self._execute_manual_trade(
                    trading_pair=trading_pair,
                    side=side,
                    amount=amount,
                    price=price
                )
                return trade_result
                
            elif action == "market_check":
                trading_pair = input_data.get("trading_pair")
                if not trading_pair:
                    return {"error": "Trading pair is required to check market conditions"}
                
                # Check market conditions
                market_conditions = await self._check_market_conditions(trading_pair)
                return {"market_conditions": market_conditions}
                
            else:
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Error in Execution Agent: {str(e)}")
            return {"error": str(e)}
    
    async def _deploy_strategy(self, strategy: Strategy) -> Dict[str, Any]:
        """Deploy a strategy for live trading.
        
        Args:
            strategy: Strategy to deploy
            
        Returns:
            Deployment status
        """
        logger.info(f"Deploying strategy: {strategy.name}")
        
        # Use execution service to deploy the strategy
        deployment_result = await execution_service.deploy_strategy(strategy)
        
        # Log deployment result
        if deployment_result.get("status") == "deployed":
            logger.info(f"Strategy {strategy.name} deployed successfully")
        else:
            logger.info(f"Strategy {strategy.name} deployment status: {deployment_result.get('status')}")
        
        return deployment_result
    
    async def _stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Stop a deployed strategy.
        
        Args:
            strategy_id: ID of the strategy to stop
            
        Returns:
            Stop status
        """
        logger.info(f"Stopping strategy: {strategy_id}")
        
        # Use execution service to stop the strategy
        stop_result = await execution_service.stop_strategy(strategy_id)
        
        # Log stop result
        logger.info(f"Strategy {strategy_id} stop status: {stop_result.get('status')}")
        
        return stop_result
    
    async def _get_strategy_performance(self, strategy_id: str) -> StrategyPerformance:
        """Get performance metrics for a deployed strategy.
        
        Args:
            strategy_id: ID of the strategy to check
            
        Returns:
            Strategy performance metrics
        """
        logger.info(f"Getting performance for strategy: {strategy_id}")
        
        # Use execution service to get performance
        performance = await execution_service.get_strategy_performance(strategy_id)
        
        # Log performance summary
        logger.info(f"Strategy {strategy_id} performance: {performance.total_return:.2%} return")
        
        return performance
    
    async def _execute_manual_trade(self, trading_pair: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Execute a manual trade.
        
        Args:
            trading_pair: Trading pair to trade
            side: Trade side (buy/sell)
            amount: Trade amount
            price: Optional limit price (market order if None)
            
        Returns:
            Trade execution result
        """
        logger.info(f"Executing manual {side} trade for {trading_pair}: {amount}")
        
        try:
            # Validate inputs
            if side not in ["buy", "sell"]:
                return {"error": f"Invalid side: {side}. Must be 'buy' or 'sell'"}
            
            if amount <= 0:
                return {"error": f"Invalid amount: {amount}. Must be positive"}
            
            # Parse trading pair to get currencies
            currencies = trading_pair.split("-")
            if len(currencies) != 2:
                return {"error": f"Invalid trading pair format: {trading_pair}. Expected format: XXX-YYY"}
            
            from_ccy = currencies[1] if side == "buy" else currencies[0]
            to_ccy = currencies[0] if side == "buy" else currencies[1]
            
            # Execute swap via OKX client
            result = await okx_client.execute_swap(
                from_ccy=from_ccy,
                to_ccy=to_ccy,
                amount=amount
            )
            
            # Check for errors
            if "error" in result:
                logger.error(f"Error executing trade: {result['error']}")
                return {"status": "failed", "error": result["error"]}
            
            # Log success
            logger.info(f"Trade executed successfully: {result}")
            
            return {
                "status": "success",
                "order_id": result.get("order_id", "unknown"),
                "trading_pair": trading_pair,
                "side": side,
                "amount": amount,
                "price": price or "market",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error executing manual trade: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def _check_market_conditions(self, trading_pair: str) -> Dict[str, Any]:
        """Check current market conditions for a trading pair.
        
        Args:
            trading_pair: Trading pair to check
            
        Returns:
            Market conditions assessment
        """
        logger.info(f"Checking market conditions for: {trading_pair}")
        
        try:
            # Get market data
            market_data = await okx_client.get_market_data(trading_pair)
            
            # Get kline data for volatility calculation
            kline_data = await okx_client.get_kline_data(
                trading_pair=trading_pair,
                bar="1h",
                limit=24  # Last 24 hours
            )
            
            # Check for errors
            if "error" in market_data or "error" in kline_data:
                logger.error(f"Error getting market data: {market_data.get('error') or kline_data.get('error')}")
                return {
                    "status": "unknown",
                    "error": market_data.get("error") or kline_data.get("error")
                }
            
            # Extract current price
            current_price = float(market_data.get("data", [{}])[0].get("last", 0))
            
            # Calculate volatility from kline data (simplified)
            prices = [float(candle[4]) for candle in kline_data.get("data", [])]  # Close prices
            if len(prices) > 1:
                returns = [(prices[i] / prices[i-1]) - 1 for i in range(1, len(prices))]
                volatility = sum([abs(r) for r in returns]) / len(returns)  # Average absolute return
            else:
                volatility = 0.0
            
            # Calculate volume
            volume = float(market_data.get("data", [{}])[0].get("vol24h", 0))
            
            # Assess market conditions
            conditions = {
                "price": current_price,
                "volatility": volatility,
                "volume": volume,
                "timestamp": time.time()
            }
            
            # Determine if conditions are favorable
            # High volatility might be unfavorable for certain strategies
            if volatility > 0.05:  # More than 5% average hourly change
                conditions["status"] = "unfavorable"
                conditions["details"] = f"High volatility detected: {volatility:.2%} average hourly change"
            # Low volume might indicate poor liquidity
            elif volume < 1000:  # Arbitrary threshold
                conditions["status"] = "unfavorable"
                conditions["details"] = f"Low trading volume detected: {volume}"
            else:
                conditions["status"] = "favorable"
                conditions["details"] = "Normal market conditions"
            
            logger.info(f"Market conditions for {trading_pair}: {conditions['status']}")
            return conditions
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {str(e)}")
            return {"status": "unknown", "error": str(e)}
    
    async def analyze_execution_quality(self, strategy_id: str) -> Dict[str, Any]:
        """Analyze the execution quality of a strategy.
        
        Args:
            strategy_id: ID of the strategy to analyze
            
        Returns:
            Execution quality analysis
        """
        logger.info(f"Analyzing execution quality for strategy: {strategy_id}")
        
        try:
            # Get strategy performance
            performance = await self._get_strategy_performance(strategy_id)
            
            # Extract recent trades
            recent_trades = performance.recent_trades
            
            # Calculate execution metrics
            if not recent_trades:
                return {
                    "status": "no_trades",
                    "message": "No recent trades to analyze"
                }
            
            # Calculate average slippage (simplified)
            # In a real implementation, would compare execution price to market price at signal time
            slippage = 0.001  # Placeholder: 0.1% average slippage
            
            # Calculate average execution time (simplified)
            # In a real implementation, would measure time from signal to execution
            execution_time = 0.5  # Placeholder: 0.5 seconds average execution time
            
            # Assess execution quality
            quality_assessment = {
                "slippage": slippage,
                "execution_time": execution_time,
                "trade_count": len(recent_trades),
                "timestamp": time.time()
            }
            
            # Determine overall quality
            if slippage > 0.005:  # More than 0.5% slippage
                quality_assessment["quality"] = "poor"
                quality_assessment["suggestions"] = ["Consider using limit orders instead of market orders"]
            elif execution_time > 2.0:  # More than 2 seconds execution time
                quality_assessment["quality"] = "fair"
                quality_assessment["suggestions"] = ["Optimize API connection for faster execution"]
            else:
                quality_assessment["quality"] = "good"
                quality_assessment["suggestions"] = ["Maintain current execution strategy"]
            
            logger.info(f"Execution quality for {strategy_id}: {quality_assessment['quality']}")
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error analyzing execution quality: {str(e)}")
            return {"status": "error", "error": str(e)}


# Create a singleton instance
execution_agent = ExecutionAgent()
