"""
Transaction Simulation Service Module

This module provides a service for simulating cryptocurrency transactions
before executing them in the real market. It allows for stress-testing
trading strategies by simulating their impact on the market under various
conditions and scenarios.

Key features:
- Simulate transactions with different parameters
- Analyze market impact of potential trades
- Stress-test strategies under various market conditions
- Generate risk reports for potential trades
- Integrate with the WebSocket system for real-time updates

This implementation helps traders and algorithms understand the potential
consequences of their trades before committing real capital.
"""

import logging
import asyncio
import json
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from app.core.okx_client import okx_client
from app.services.websocket_service import MessagePublisher
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SimulationRequest(BaseModel):
    """Model for a transaction simulation request."""
    trading_pair: str
    side: str  # "buy" or "sell"
    amount: float
    price: Optional[float] = None  # If None, use market price
    slippage_model: str = "linear"  # "linear", "exponential", "historical"
    market_conditions: str = "normal"  # "normal", "volatile", "trending", "illiquid"
    time_horizon: str = "immediate"  # "immediate", "day", "week"
    confidence_level: float = 0.95  # For VaR calculations
    num_simulations: int = 1000  # Number of Monte Carlo simulations
    strategy_id: Optional[str] = None  # ID of the strategy being tested


class SimulationResult(BaseModel):
    """Model for a transaction simulation result."""
    request_id: str
    trading_pair: str
    side: str
    amount: float
    initial_price: float
    expected_execution_price: float
    price_impact: float  # Percentage
    slippage: float  # Percentage
    expected_cost: float
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    max_drawdown: float  # Maximum potential drawdown
    liquidity_score: float  # 0-100 score of market liquidity
    execution_probability: float  # Probability of successful execution
    warning_flags: List[str]  # Any warning flags
    timestamp: str


class TransactionSimulationService:
    """Service for simulating cryptocurrency transactions and stress-testing strategies."""
    
    def __init__(self, websocket_channel: Optional[str] = None):
        """Initialize the transaction simulation service.
        
        Args:
            websocket_channel: Optional channel for publishing simulation updates
        """
        self.websocket_channel = websocket_channel
        self.simulation_history = []  # Store simulation history
        self.market_data_cache = {}  # Cache market data to reduce API calls
    
    async def simulate_transaction(self, request: SimulationRequest) -> Dict[str, Any]:
        """Simulate a transaction to analyze market impact.
        
        Args:
            request: Simulation request parameters
            
        Returns:
            Simulation results
        """
        try:
            logger.info(f"Simulating transaction: {request.trading_pair}, {request.side}, {request.amount}")
            
            # Generate a request ID
            request_id = f"sim_{int(time.time() * 1000)}"
            
            # Get current market data
            market_data = await self._get_market_data(request.trading_pair)
            if "error" in market_data:
                return {"status": "failed", "error": market_data["error"]}
            
            # Get current price
            current_price = market_data["last_price"]
            initial_price = request.price or current_price
            
            # Calculate expected execution price based on slippage model
            expected_price, price_impact = await self._calculate_price_impact(
                request.trading_pair,
                request.side,
                request.amount,
                initial_price,
                request.slippage_model,
                request.market_conditions
            )
            
            # Calculate slippage
            slippage = abs(expected_price - initial_price) / initial_price * 100
            
            # Calculate expected cost
            expected_cost = request.amount * expected_price
            
            # Perform risk analysis
            risk_metrics = await self._calculate_risk_metrics(
                request.trading_pair,
                request.side,
                request.amount,
                expected_price,
                request.confidence_level,
                request.num_simulations,
                request.market_conditions
            )
            
            # Determine warning flags
            warning_flags = self._determine_warning_flags(
                price_impact,
                slippage,
                risk_metrics["liquidity_score"],
                risk_metrics["execution_probability"],
                market_data
            )
            
            # Create simulation result
            result = SimulationResult(
                request_id=request_id,
                trading_pair=request.trading_pair,
                side=request.side,
                amount=request.amount,
                initial_price=initial_price,
                expected_execution_price=expected_price,
                price_impact=price_impact,
                slippage=slippage,
                expected_cost=expected_cost,
                var_95=risk_metrics["var_95"],
                var_99=risk_metrics["var_99"],
                max_drawdown=risk_metrics["max_drawdown"],
                liquidity_score=risk_metrics["liquidity_score"],
                execution_probability=risk_metrics["execution_probability"],
                warning_flags=warning_flags,
                timestamp=datetime.now().isoformat()
            )
            
            # Store in simulation history
            self.simulation_history.append({
                "request_id": request_id,
                "request": request.dict(),
                "result": result.dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Publish result to WebSocket channel
            if self.websocket_channel:
                await self._publish_simulation_result(result.dict())
            
            return {
                "status": "success",
                "request_id": request_id,
                "result": result.dict()
            }
            
        except Exception as e:
            logger.error(f"Error simulating transaction: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def stress_test_strategy(self, 
                                  strategy_id: str, 
                                  trading_pairs: List[str],
                                  position_sizes: List[float],
                                  market_scenarios: List[str] = None,
                                  num_simulations: int = 100) -> Dict[str, Any]:
        """Stress test a strategy under various market conditions.
        
        Args:
            strategy_id: ID of the strategy to test
            trading_pairs: List of trading pairs to test
            position_sizes: List of position sizes to test (as percentage of portfolio)
            market_scenarios: List of market scenarios to test
            num_simulations: Number of simulations to run
            
        Returns:
            Stress test results
        """
        try:
            logger.info(f"Stress testing strategy {strategy_id}")
            
            # Default market scenarios if not provided
            if not market_scenarios:
                market_scenarios = ["normal", "volatile", "trending", "illiquid"]
            
            # Initialize results
            results = {
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat(),
                "overall_risk_score": 0,
                "scenarios": [],
                "recommendations": []
            }
            
            # Run simulations for each combination of parameters
            total_risk_score = 0
            scenario_count = 0
            
            for trading_pair in trading_pairs:
                for position_size in position_sizes:
                    for scenario in market_scenarios:
                        # Create simulation request for buy side
                        buy_request = SimulationRequest(
                            trading_pair=trading_pair,
                            side="buy",
                            amount=position_size,
                            market_conditions=scenario,
                            num_simulations=num_simulations,
                            strategy_id=strategy_id
                        )
                        
                        # Run buy simulation
                        buy_result = await self.simulate_transaction(buy_request)
                        
                        # Create simulation request for sell side
                        sell_request = SimulationRequest(
                            trading_pair=trading_pair,
                            side="sell",
                            amount=position_size,
                            market_conditions=scenario,
                            num_simulations=num_simulations,
                            strategy_id=strategy_id
                        )
                        
                        # Run sell simulation
                        sell_result = await self.simulate_transaction(sell_request)
                        
                        # Calculate scenario risk score
                        if buy_result["status"] == "success" and sell_result["status"] == "success":
                            buy_data = buy_result["result"]
                            sell_data = sell_result["result"]
                            
                            # Calculate risk score for this scenario
                            risk_score = self._calculate_scenario_risk_score(buy_data, sell_data, scenario)
                            total_risk_score += risk_score
                            scenario_count += 1
                            
                            # Add scenario result
                            results["scenarios"].append({
                                "trading_pair": trading_pair,
                                "position_size": position_size,
                                "market_condition": scenario,
                                "risk_score": risk_score,
                                "buy_simulation": buy_data,
                                "sell_simulation": sell_data
                            })
            
            # Calculate overall risk score
            if scenario_count > 0:
                results["overall_risk_score"] = total_risk_score / scenario_count
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)
            
            # Publish results to WebSocket channel
            if self.websocket_channel:
                await MessagePublisher.publish(
                    self.websocket_channel,
                    {
                        "type": "stress_test_results",
                        "data": results
                    }
                )
            
            return {
                "status": "success",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error stress testing strategy: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_simulation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get transaction simulation history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            Simulation history
        """
        return self.simulation_history[-limit:]
    
    async def _get_market_data(self, trading_pair: str) -> Dict[str, Any]:
        """Get current market data for a trading pair.
        
        Args:
            trading_pair: Trading pair to get data for
            
        Returns:
            Market data
        """
        try:
            # Check cache first (cache for 60 seconds)
            cache_key = f"{trading_pair}_market_data"
            if cache_key in self.market_data_cache:
                cache_time, cache_data = self.market_data_cache[cache_key]
                if datetime.now() - cache_time < timedelta(seconds=60):
                    return cache_data
            
            # Fetch from API
            response = await okx_client.get_market_data(trading_pair)
            
            # Check for errors
            if "error" in response:
                return {"error": response["error"]}
            
            # Extract data
            ticker_data = response.get("data", [{}])[0]
            
            # Process data
            market_data = {
                "trading_pair": trading_pair,
                "last_price": float(ticker_data.get("last", 0)),
                "bid_price": float(ticker_data.get("bidPx", 0)),
                "ask_price": float(ticker_data.get("askPx", 0)),
                "volume_24h": float(ticker_data.get("vol24h", 0)),
                "open_24h": float(ticker_data.get("open24h", 0)),
                "high_24h": float(ticker_data.get("high24h", 0)),
                "low_24h": float(ticker_data.get("low24h", 0)),
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate volatility
            if market_data["high_24h"] > 0 and market_data["low_24h"] > 0:
                market_data["volatility_24h"] = (market_data["high_24h"] - market_data["low_24h"]) / market_data["low_24h"] * 100
            else:
                market_data["volatility_24h"] = 0
            
            # Calculate spread
            if market_data["ask_price"] > 0:
                market_data["spread"] = (market_data["ask_price"] - market_data["bid_price"]) / market_data["ask_price"] * 100
            else:
                market_data["spread"] = 0
            
            # Cache the data
            self.market_data_cache[cache_key] = (datetime.now(), market_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return {"error": str(e)}
    
    async def _calculate_price_impact(self, 
                                    trading_pair: str, 
                                    side: str, 
                                    amount: float, 
                                    initial_price: float,
                                    slippage_model: str,
                                    market_conditions: str) -> tuple:
        """Calculate expected price impact of a transaction.
        
        Args:
            trading_pair: Trading pair
            side: "buy" or "sell"
            amount: Amount to trade
            initial_price: Initial price
            slippage_model: Slippage model to use
            market_conditions: Market conditions to simulate
            
        Returns:
            Tuple of (expected_price, price_impact_percentage)
        """
        # Get market data
        market_data = await self._get_market_data(trading_pair)
        if "error" in market_data:
            # Default to 0.1% impact if we can't get market data
            expected_price = initial_price * (1.001 if side == "buy" else 0.999)
            return expected_price, 0.1
        
        # Get 24h volume
        volume_24h = market_data["volume_24h"]
        
        # Calculate base impact factor based on size relative to 24h volume
        # Larger trades relative to volume = higher impact
        volume_factor = min(1.0, amount / (volume_24h * 0.01))  # Cap at 1.0 (100%)
        
        # Adjust for market conditions
        condition_factors = {
            "normal": 1.0,
            "volatile": 2.0,
            "trending": 1.5,
            "illiquid": 3.0
        }
        condition_factor = condition_factors.get(market_conditions, 1.0)
        
        # Calculate impact based on selected model
        if slippage_model == "linear":
            # Linear impact: directly proportional to size
            impact_percentage = volume_factor * 0.2 * condition_factor  # Base 0.2% impact for 1% of daily volume
            
        elif slippage_model == "exponential":
            # Exponential impact: grows exponentially with size
            impact_percentage = (volume_factor ** 1.5) * 0.2 * condition_factor
            
        elif slippage_model == "historical":
            # Historical impact: based on spread and volatility
            spread = market_data["spread"]
            volatility = market_data["volatility_24h"]
            impact_percentage = (spread * 2 + volatility * 0.05) * volume_factor * condition_factor
            
        else:
            # Default to linear
            impact_percentage = volume_factor * 0.2 * condition_factor
        
        # Cap the impact at reasonable levels
        impact_percentage = min(10.0, impact_percentage)  # Cap at 10%
        
        # Calculate expected price
        if side == "buy":
            expected_price = initial_price * (1 + impact_percentage / 100)
        else:  # sell
            expected_price = initial_price * (1 - impact_percentage / 100)
        
        return expected_price, impact_percentage
    
    async def _calculate_risk_metrics(self,
                                    trading_pair: str,
                                    side: str,
                                    amount: float,
                                    expected_price: float,
                                    confidence_level: float,
                                    num_simulations: int,
                                    market_conditions: str) -> Dict[str, Any]:
        """Calculate risk metrics for a transaction.
        
        Args:
            trading_pair: Trading pair
            side: "buy" or "sell"
            amount: Amount to trade
            expected_price: Expected execution price
            confidence_level: Confidence level for VaR
            num_simulations: Number of Monte Carlo simulations
            market_conditions: Market conditions to simulate
            
        Returns:
            Risk metrics
        """
        # Get market data
        market_data = await self._get_market_data(trading_pair)
        if "error" in market_data:
            # Return default values if we can't get market data
            return {
                "var_95": amount * expected_price * 0.05,
                "var_99": amount * expected_price * 0.08,
                "max_drawdown": amount * expected_price * 0.1,
                "liquidity_score": 50.0,
                "execution_probability": 0.9
            }
        
        # Get market parameters
        volatility = market_data["volatility_24h"] / 100  # Convert to decimal
        spread = market_data["spread"] / 100  # Convert to decimal
        volume = market_data["volume_24h"]
        
        # Adjust volatility based on market conditions
        volatility_factors = {
            "normal": 1.0,
            "volatile": 2.5,
            "trending": 1.2,
            "illiquid": 2.0
        }
        adjusted_volatility = volatility * volatility_factors.get(market_conditions, 1.0)
        
        # Run Monte Carlo simulations
        price_paths = []
        for _ in range(num_simulations):
            # Simulate price path
            price_change = np.random.normal(0, adjusted_volatility)
            simulated_price = expected_price * (1 + price_change)
            price_paths.append(simulated_price)
        
        # Calculate VaR
        price_changes = [(p - expected_price) / expected_price for p in price_paths]
        price_changes.sort()
        
        # 95% VaR
        var_95_index = int(num_simulations * 0.05)
        var_95_pct = abs(price_changes[var_95_index])
        var_95 = amount * expected_price * var_95_pct
        
        # 99% VaR
        var_99_index = int(num_simulations * 0.01)
        var_99_pct = abs(price_changes[var_99_index])
        var_99 = amount * expected_price * var_99_pct
        
        # Calculate max drawdown
        max_drawdown_pct = abs(min(price_changes))
        max_drawdown = amount * expected_price * max_drawdown_pct
        
        # Calculate liquidity score (0-100)
        # Based on volume, spread, and volatility
        volume_score = min(100, volume / 1000)  # Normalize volume
        spread_score = max(0, 100 - spread * 10000)  # Lower spread = higher score
        volatility_score = max(0, 100 - adjusted_volatility * 1000)  # Lower volatility = higher score
        
        liquidity_score = (volume_score * 0.5 + spread_score * 0.3 + volatility_score * 0.2)
        
        # Calculate execution probability
        # Higher for liquid markets, lower for illiquid or volatile markets
        base_probability = 0.99  # Base probability
        volume_factor = min(1.0, amount / (volume * 0.01))  # Size relative to volume
        execution_probability = base_probability - (volume_factor * 0.2) - (adjusted_volatility * 0.5)
        execution_probability = max(0.5, min(0.99, execution_probability))  # Clamp between 0.5 and 0.99
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "max_drawdown": max_drawdown,
            "liquidity_score": liquidity_score,
            "execution_probability": execution_probability
        }
    
    def _determine_warning_flags(self,
                               price_impact: float,
                               slippage: float,
                               liquidity_score: float,
                               execution_probability: float,
                               market_data: Dict[str, Any]) -> List[str]:
        """Determine warning flags for a transaction.
        
        Args:
            price_impact: Price impact percentage
            slippage: Slippage percentage
            liquidity_score: Liquidity score
            execution_probability: Execution probability
            market_data: Market data
            
        Returns:
            List of warning flags
        """
        warnings = []
        
        # Check price impact
        if price_impact > 5.0:
            warnings.append("high_price_impact")
        elif price_impact > 2.0:
            warnings.append("moderate_price_impact")
        
        # Check slippage
        if slippage > 3.0:
            warnings.append("high_slippage")
        elif slippage > 1.0:
            warnings.append("moderate_slippage")
        
        # Check liquidity
        if liquidity_score < 30:
            warnings.append("very_low_liquidity")
        elif liquidity_score < 50:
            warnings.append("low_liquidity")
        
        # Check execution probability
        if execution_probability < 0.7:
            warnings.append("low_execution_probability")
        
        # Check volatility
        volatility = market_data.get("volatility_24h", 0)
        if volatility > 10:
            warnings.append("high_volatility")
        elif volatility > 5:
            warnings.append("moderate_volatility")
        
        # Check spread
        spread = market_data.get("spread", 0)
        if spread > 1.0:
            warnings.append("wide_spread")
        
        return warnings
    
    def _calculate_scenario_risk_score(self, 
                                     buy_data: Dict[str, Any], 
                                     sell_data: Dict[str, Any],
                                     scenario: str) -> float:
        """Calculate risk score for a scenario.
        
        Args:
            buy_data: Buy simulation data
            sell_data: Sell simulation data
            scenario: Market scenario
            
        Returns:
            Risk score (0-100, higher = riskier)
        """
        # Extract metrics
        buy_impact = buy_data["price_impact"]
        sell_impact = sell_data["price_impact"]
        buy_var = buy_data["var_95"]
        sell_var = sell_data["var_95"]
        buy_liquidity = buy_data["liquidity_score"]
        sell_liquidity = sell_data["liquidity_score"]
        buy_probability = buy_data["execution_probability"]
        sell_probability = sell_data["execution_probability"]
        
        # Calculate average metrics
        avg_impact = (buy_impact + sell_impact) / 2
        avg_liquidity = (buy_liquidity + sell_liquidity) / 2
        avg_probability = (buy_probability + sell_probability) / 2
        
        # Calculate risk components
        impact_risk = min(100, avg_impact * 10)  # 0-100
        liquidity_risk = max(0, 100 - avg_liquidity)  # 0-100
        execution_risk = max(0, (1 - avg_probability) * 100)  # 0-100
        
        # Adjust weights based on scenario
        if scenario == "normal":
            weights = {
                "impact": 0.3,
                "liquidity": 0.3,
                "execution": 0.4
            }
        elif scenario == "volatile":
            weights = {
                "impact": 0.5,
                "liquidity": 0.2,
                "execution": 0.3
            }
        elif scenario == "trending":
            weights = {
                "impact": 0.4,
                "liquidity": 0.3,
                "execution": 0.3
            }
        elif scenario == "illiquid":
            weights = {
                "impact": 0.3,
                "liquidity": 0.5,
                "execution": 0.2
            }
        else:
            weights = {
                "impact": 0.33,
                "liquidity": 0.33,
                "execution": 0.34
            }
        
        # Calculate weighted risk score
        risk_score = (
            impact_risk * weights["impact"] +
            liquidity_risk * weights["liquidity"] +
            execution_risk * weights["execution"]
        )
        
        return risk_score
    
    def _generate_recommendations(self, stress_test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stress test results.
        
        Args:
            stress_test_results: Stress test results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        overall_risk = stress_test_results["overall_risk_score"]
        scenarios = stress_test_results["scenarios"]
        
        # Overall risk recommendations
        if overall_risk > 75:
            recommendations.append("Consider reducing position sizes significantly due to high overall risk.")
        elif overall_risk > 50:
            recommendations.append("Consider reducing position sizes or implementing stronger risk controls.")
        elif overall_risk > 25:
            recommendations.append("Current risk levels are moderate. Regular monitoring is advised.")
        else:
            recommendations.append("Risk levels are acceptable. Strategy appears robust across tested scenarios.")
        
        # Analyze scenarios for specific recommendations
        high_risk_scenarios = [s for s in scenarios if s["risk_score"] > 70]
        if high_risk_scenarios:
            high_risk_pairs = set(s["trading_pair"] for s in high_risk_scenarios)
            high_risk_conditions = set(s["market_condition"] for s in high_risk_scenarios)
            
            if len(high_risk_pairs) > 0:
                pairs_str = ", ".join(high_risk_pairs)
                recommendations.append(f"Consider reducing exposure to {pairs_str} which showed high risk.")
            
            if "illiquid" in high_risk_conditions:
                recommendations.append("Strategy is particularly vulnerable to illiquid market conditions. Consider implementing liquidity checks before execution.")
            
            if "volatile" in high_risk_conditions:
                recommendations.append("Strategy is sensitive to market volatility. Consider implementing volatility-based position sizing.")
        
        # Position size recommendations
        position_sizes = [s["position_size"] for s in scenarios]
        risk_by_size = {}
        for s in scenarios:
            size = s["position_size"]
            if size not in risk_by_size:
                risk_by_size[size] = []
            risk_by_size[size].append(s["risk_score"])
        
        avg_risk_by_size = {size: sum(scores)/len(scores) for size, scores in risk_by_size.items()}
        optimal_size = min(avg_risk_by_size.items(), key=lambda x: x[1])[0]
        
        if optimal_size < max(position_sizes):
            recommendations.append(f"Optimal position size appears to be around {optimal_size} based on risk-adjusted performance.")
        
        # Add general recommendations
        recommendations.append("Implement circuit breakers to automatically halt trading during extreme market conditions.")
        recommendations.append("Consider using time-weighted average price (TWAP) or volume-weighted average price (VWAP) algorithms for large orders.")
        
        return recommendations
    
    async def _publish_simulation_result(self, result: Dict[str, Any]):
        """Publish simulation result to WebSocket channel.
        
        Args:
            result: Simulation result
        """
        if not self.websocket_channel:
            return
        
        message = {
            "type": "transaction_simulation",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
        await MessagePublisher.publish(self.websocket_channel, message)


# Create a singleton instance
transaction_simulation_service = TransactionSimulationService(websocket_channel="transaction_simulation")
