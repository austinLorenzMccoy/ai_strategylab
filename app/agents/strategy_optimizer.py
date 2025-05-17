import logging
import time
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from app.agents.base_agent import BaseAgent, AgentAction, AgentObservation
from app.models.strategy import Strategy, BacktestResult
from app.services.rl_service import rl_service
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StrategyOptimizerAgent(BaseAgent):
    """Agent responsible for optimizing trading strategies using RL."""
    
    def __init__(self):
        """Initialize the strategy optimizer agent."""
        super().__init__(name="Strategy Optimizer Agent")
        
        # Initialize LLM if API key is available
        if settings.OPENAI_API_KEY:
            self.chat_model = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.LLM_MODEL,
                temperature=0.2
            )
        
        # Register tools
        self.register_tool(
            name="optimize_strategy",
            func=self._optimize_strategy,
            description="Optimize a trading strategy using reinforcement learning"
        )
        self.register_tool(
            name="backtest_strategy",
            func=self._backtest_strategy,
            description="Backtest a trading strategy against historical data"
        )
        self.register_tool(
            name="analyze_strategy_performance",
            func=self._analyze_strategy_performance,
            description="Analyze the performance of a strategy and suggest improvements"
        )
        
        # System prompt
        self.system_prompt = """
        You are the Strategy Optimizer Agent, an AI specialized in optimizing trading strategies.
        Your primary responsibilities are:
        1. Optimize strategy parameters using reinforcement learning
        2. Backtest strategies against historical data
        3. Analyze performance metrics and suggest improvements
        4. Balance risk and reward based on the user's risk tolerance
        
        You have expertise in technical indicators, market patterns, and optimization techniques.
        Always consider the risk tolerance specified in the strategy when making optimization decisions.
        """
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent on input data.
        
        Args:
            input_data: Input data containing strategy and historical data
            
        Returns:
            Optimization results
        """
        try:
            logger.info("Running Strategy Optimizer Agent")
            
            # Extract input data
            strategy = input_data.get("strategy")
            historical_data = input_data.get("historical_data")
            action = input_data.get("action", "optimize")
            
            # Validate input
            if not strategy:
                return {"error": "Strategy is required"}
            
            if not historical_data and (action == "optimize" or action == "backtest"):
                return {"error": "Historical data is required for optimization and backtesting"}
            
            # Determine action
            if action == "optimize":
                # Optimize the strategy
                optimized_strategy = await self._optimize_strategy(
                    strategy=strategy,
                    historical_data=historical_data
                )
                
                # Backtest the optimized strategy
                backtest_result = await self._backtest_strategy(
                    strategy=optimized_strategy,
                    historical_data=historical_data
                )
                
                # Analyze performance
                analysis = await self._analyze_strategy_performance(
                    strategy=optimized_strategy,
                    backtest_result=backtest_result
                )
                
                return {
                    "optimized_strategy": optimized_strategy,
                    "backtest_result": backtest_result,
                    "analysis": analysis
                }
                
            elif action == "backtest":
                # Backtest the strategy
                backtest_result = await self._backtest_strategy(
                    strategy=strategy,
                    historical_data=historical_data
                )
                
                # Analyze performance
                analysis = await self._analyze_strategy_performance(
                    strategy=strategy,
                    backtest_result=backtest_result
                )
                
                return {
                    "backtest_result": backtest_result,
                    "analysis": analysis
                }
                
            elif action == "analyze":
                # Extract backtest result
                backtest_result = input_data.get("backtest_result")
                if not backtest_result:
                    return {"error": "Backtest result is required for analysis"}
                
                # Analyze performance
                analysis = await self._analyze_strategy_performance(
                    strategy=strategy,
                    backtest_result=backtest_result
                )
                
                return {"analysis": analysis}
                
            else:
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Error in Strategy Optimizer Agent: {str(e)}")
            return {"error": str(e)}
    
    async def _optimize_strategy(self, strategy: Strategy, historical_data: List[Dict[str, Any]]) -> Strategy:
        """Optimize a strategy using reinforcement learning.
        
        Args:
            strategy: Strategy to optimize
            historical_data: Historical market data
            
        Returns:
            Optimized strategy
        """
        logger.info(f"Optimizing strategy: {strategy.name}")
        
        # Use RL service to optimize the strategy
        optimized_strategy = await rl_service.optimize_strategy(strategy, historical_data)
        
        # Log optimization results
        logger.info(f"Strategy optimization complete for {strategy.name}")
        
        return optimized_strategy
    
    async def _backtest_strategy(self, strategy: Strategy, historical_data: List[Dict[str, Any]]) -> BacktestResult:
        """Backtest a strategy against historical data.
        
        Args:
            strategy: Strategy to backtest
            historical_data: Historical market data
            
        Returns:
            Backtest results
        """
        logger.info(f"Backtesting strategy: {strategy.name}")
        
        # Use RL service to backtest the strategy
        backtest_result = await rl_service.backtest_strategy(strategy, historical_data)
        
        # Log backtest results
        logger.info(f"Backtest complete for {strategy.name} with return: {backtest_result.total_return:.2%}")
        
        return backtest_result
    
    async def _analyze_strategy_performance(self, strategy: Strategy, backtest_result: BacktestResult) -> Dict[str, Any]:
        """Analyze strategy performance and suggest improvements.
        
        Args:
            strategy: Strategy to analyze
            backtest_result: Backtest results
            
        Returns:
            Performance analysis and suggestions
        """
        logger.info(f"Analyzing performance for strategy: {strategy.name}")
        
        # Prepare analysis
        analysis = {
            "performance_summary": {
                "total_return": backtest_result.total_return,
                "sharpe_ratio": backtest_result.sharpe_ratio,
                "max_drawdown": backtest_result.max_drawdown,
                "win_rate": backtest_result.win_rate,
                "num_trades": backtest_result.num_trades
            },
            "risk_assessment": self._assess_risk(backtest_result, strategy.risk_tolerance),
            "suggestions": []
        }
        
        # Generate suggestions using LLM if available
        if self.chat_model:
            try:
                # Prepare prompt for LLM
                prompt = f"""
                Analyze the following trading strategy performance and provide suggestions for improvement:
                
                Strategy: {strategy.name}
                Type: {strategy.type.value}
                Risk Tolerance: {strategy.risk_tolerance.value}
                
                Performance Metrics:
                - Total Return: {backtest_result.total_return:.2%}
                - Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}
                - Max Drawdown: {backtest_result.max_drawdown:.2%}
                - Win Rate: {backtest_result.win_rate:.2%}
                - Number of Trades: {backtest_result.num_trades}
                
                Current Indicators:
                {[ind.name for ind in strategy.indicators]}
                
                Current Rules:
                {[rule.name for rule in strategy.rules]}
                
                Please provide 3-5 specific suggestions to improve this strategy's performance.
                Focus on risk management, entry/exit timing, and parameter tuning.
                """
                
                # Call LLM for suggestions
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                llm_response = await self._call_chat_model(messages)
                
                # Parse suggestions
                suggestions = []
                for line in llm_response.split("\n"):
                    line = line.strip()
                    if line and (line.startswith("-") or line.startswith("1.") or line.startswith("2.")):
                        suggestions.append(line.lstrip("- 123456789.").strip())
                
                analysis["suggestions"] = suggestions[:5]  # Limit to 5 suggestions
                
            except Exception as e:
                logger.error(f"Error generating suggestions: {str(e)}")
                analysis["suggestions"] = ["Error generating suggestions"]
        else:
            # Default suggestions if LLM is not available
            if backtest_result.max_drawdown > 0.2:
                analysis["suggestions"].append("Consider adding stop-loss to reduce maximum drawdown")
            
            if backtest_result.win_rate < 0.4:
                analysis["suggestions"].append("Adjust entry conditions to improve win rate")
            
            if backtest_result.sharpe_ratio < 1.0:
                analysis["suggestions"].append("Optimize risk-reward ratio to improve Sharpe ratio")
            
            if backtest_result.num_trades < 10:
                analysis["suggestions"].append("Consider relaxing conditions to increase trading frequency")
        
        logger.info(f"Analysis complete for {strategy.name}")
        return analysis
    
    def _assess_risk(self, backtest_result: BacktestResult, risk_tolerance: str) -> Dict[str, Any]:
        """Assess risk based on backtest results and risk tolerance.
        
        Args:
            backtest_result: Backtest results
            risk_tolerance: Risk tolerance level
            
        Returns:
            Risk assessment
        """
        # Define risk thresholds based on risk tolerance
        if risk_tolerance == "low":
            max_drawdown_threshold = 0.1  # 10%
            volatility_threshold = 0.02   # 2% daily
        elif risk_tolerance == "medium":
            max_drawdown_threshold = 0.2  # 20%
            volatility_threshold = 0.03   # 3% daily
        else:  # high
            max_drawdown_threshold = 0.3  # 30%
            volatility_threshold = 0.05   # 5% daily
        
        # Calculate volatility from returns (simplified)
        volatility = 0.03  # Placeholder, would calculate from actual returns
        
        # Assess risk
        risk_level = "low"
        risk_factors = []
        
        if backtest_result.max_drawdown > max_drawdown_threshold:
            risk_level = "high"
            risk_factors.append(f"Maximum drawdown ({backtest_result.max_drawdown:.2%}) exceeds threshold for {risk_tolerance} risk tolerance")
        
        if volatility > volatility_threshold:
            risk_level = "high"
            risk_factors.append(f"Volatility ({volatility:.2%}) exceeds threshold for {risk_tolerance} risk tolerance")
        
        if backtest_result.sharpe_ratio < 1.0:
            risk_level = "medium" if risk_level != "high" else risk_level
            risk_factors.append(f"Low Sharpe ratio ({backtest_result.sharpe_ratio:.2f}) indicates poor risk-adjusted returns")
        
        # If no risk factors, set risk level based on risk tolerance
        if not risk_factors:
            risk_level = risk_tolerance
            risk_factors.append(f"No significant risk factors identified, maintaining {risk_tolerance} risk profile")
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "within_tolerance": risk_level != "high" or risk_tolerance == "high"
        }


# Create a singleton instance
strategy_optimizer_agent = StrategyOptimizerAgent()
