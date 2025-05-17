import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from app.models.strategy import Strategy, StrategyPerformance
from app.core.config import get_settings
from app.core.okx_client import okx_client

logger = logging.getLogger(__name__)
settings = get_settings()


class ExecutionService:
    """Service for executing trading strategies."""
    
    def __init__(self):
        """Initialize the execution service."""
        self.active_strategies = {}  # Store active strategies by ID
        self.strategy_tasks = {}  # Store asyncio tasks for each strategy
    
    async def deploy_strategy(self, strategy: Strategy) -> Dict[str, Any]:
        """Deploy a strategy for live trading.
        
        Args:
            strategy: The strategy to deploy
            
        Returns:
            Deployment status
        """
        try:
            strategy_id = strategy.id
            if not strategy_id:
                raise ValueError("Strategy ID is required for deployment")
                
            logger.info(f"Deploying strategy {strategy_id} for trading")
            
            # Check if strategy is already deployed
            if strategy_id in self.active_strategies:
                logger.warning(f"Strategy {strategy_id} is already deployed")
                return {"status": "already_deployed", "strategy_id": strategy_id}
            
            # Store strategy in active strategies
            self.active_strategies[strategy_id] = {
                "strategy": strategy,
                "status": "starting",
                "start_time": time.time(),
                "last_update": time.time(),
                "positions": {},
                "trades": [],
                "performance": {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "num_trades": 0
                }
            }
            
            # Start strategy execution task
            task = asyncio.create_task(self._run_strategy(strategy_id))
            self.strategy_tasks[strategy_id] = task
            
            logger.info(f"Strategy {strategy_id} deployed successfully")
            return {"status": "deployed", "strategy_id": strategy_id}
            
        except Exception as e:
            logger.error(f"Error deploying strategy: {str(e)}")
            raise
    
    async def stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Stop a deployed strategy.
        
        Args:
            strategy_id: ID of the strategy to stop
            
        Returns:
            Stop status
        """
        try:
            logger.info(f"Stopping strategy {strategy_id}")
            
            # Check if strategy is deployed
            if strategy_id not in self.active_strategies:
                logger.warning(f"Strategy {strategy_id} is not deployed")
                return {"status": "not_deployed", "strategy_id": strategy_id}
            
            # Cancel strategy task
            if strategy_id in self.strategy_tasks:
                task = self.strategy_tasks[strategy_id]
                if not task.done():
                    task.cancel()
                del self.strategy_tasks[strategy_id]
            
            # Update strategy status
            self.active_strategies[strategy_id]["status"] = "stopped"
            
            logger.info(f"Strategy {strategy_id} stopped successfully")
            return {"status": "stopped", "strategy_id": strategy_id}
            
        except Exception as e:
            logger.error(f"Error stopping strategy: {str(e)}")
            raise
    
    async def get_strategy_performance(self, strategy_id: str) -> StrategyPerformance:
        """Get performance metrics for a deployed strategy.
        
        Args:
            strategy_id: ID of the strategy to check
            
        Returns:
            Strategy performance metrics
        """
        try:
            logger.info(f"Getting performance for strategy {strategy_id}")
            
            # Check if strategy is deployed
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Strategy {strategy_id} is not deployed")
            
            # Get strategy data
            strategy_data = self.active_strategies[strategy_id]
            
            # Create performance object
            performance = StrategyPerformance(
                strategy_id=strategy_id,
                start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(strategy_data["start_time"])),
                current_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                total_return=strategy_data["performance"]["total_return"],
                current_positions=strategy_data["positions"],
                recent_trades=strategy_data["trades"][-10:] if strategy_data["trades"] else [],
                metrics=strategy_data["performance"]
            )
            
            logger.info(f"Retrieved performance for strategy {strategy_id}")
            return performance
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {str(e)}")
            raise
    
    async def _run_strategy(self, strategy_id: str):
        """Run a strategy continuously.
        
        Args:
            strategy_id: ID of the strategy to run
        """
        try:
            logger.info(f"Starting execution of strategy {strategy_id}")
            
            # Get strategy data
            strategy_data = self.active_strategies[strategy_id]
            strategy = strategy_data["strategy"]
            
            # Update status
            strategy_data["status"] = "running"
            
            # Main execution loop
            while True:
                try:
                    # Update last update time
                    strategy_data["last_update"] = time.time()
                    
                    # Get market data for all trading pairs
                    market_data = {}
                    for trading_pair in strategy.trading_pairs:
                        # Use OKX client to get market data
                        pair_data = await okx_client.get_market_data(trading_pair)
                        market_data[trading_pair] = pair_data
                    
                    # Evaluate strategy rules
                    signals = await self._evaluate_strategy_rules(strategy, market_data)
                    
                    # Execute signals
                    for trading_pair, signal in signals.items():
                        if signal != 0:  # 0 means no action
                            await self._execute_signal(strategy_id, trading_pair, signal, market_data[trading_pair])
                    
                    # Sleep before next iteration
                    await asyncio.sleep(settings.STRATEGY_EXECUTION_INTERVAL)  # e.g., 60 seconds
                    
                except asyncio.CancelledError:
                    logger.info(f"Strategy {strategy_id} execution cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in strategy {strategy_id} execution loop: {str(e)}")
                    # Sleep before retry
                    await asyncio.sleep(5)
            
            # Update status when exiting
            strategy_data["status"] = "stopped"
            logger.info(f"Strategy {strategy_id} execution stopped")
            
        except Exception as e:
            logger.error(f"Error running strategy {strategy_id}: {str(e)}")
            # Update status on error
            if strategy_id in self.active_strategies:
                self.active_strategies[strategy_id]["status"] = "error"
    
    async def _evaluate_strategy_rules(self, strategy: Strategy, market_data: Dict[str, Any]) -> Dict[str, int]:
        """Evaluate strategy rules to generate trading signals.
        
        Args:
            strategy: The strategy to evaluate
            market_data: Current market data for all trading pairs
            
        Returns:
            Dictionary of trading signals by trading pair (1: buy, -1: sell, 0: hold)
        """
        signals = {pair: 0 for pair in strategy.trading_pairs}  # Default to hold
        
        # Process each rule
        for rule in strategy.rules:
            # Check if all conditions are met
            conditions_met = True
            
            for condition in rule.conditions:
                # Get indicator value from market data
                # In a real implementation, calculate indicators from market data
                indicator_name = condition.indicator
                indicator_value = market_data.get(strategy.trading_pairs[0], {}).get(indicator_name.lower(), 50)  # Default value
                
                # Compare with condition
                if condition.operator == "<":
                    if not (indicator_value < condition.value):
                        conditions_met = False
                        break
                elif condition.operator == ">":
                    if not (indicator_value > condition.value):
                        conditions_met = False
                        break
                elif condition.operator == "==":
                    if not (indicator_value == condition.value):
                        conditions_met = False
                        break
            
            # If all conditions are met, apply actions
            if conditions_met:
                for action in rule.actions:
                    trading_pair = next((pair for pair in strategy.trading_pairs if pair.startswith(action.asset)), None)
                    if trading_pair:
                        if action.type == "BUY":
                            signals[trading_pair] = 1
                        elif action.type == "SELL":
                            signals[trading_pair] = -1
        
        return signals
    
    async def _execute_signal(self, strategy_id: str, trading_pair: str, signal: int, market_data: Dict[str, Any]):
        """Execute a trading signal.
        
        Args:
            strategy_id: ID of the strategy
            trading_pair: Trading pair to trade
            signal: Trading signal (1: buy, -1: sell)
            market_data: Current market data for the trading pair
        """
        try:
            # Get strategy data
            strategy_data = self.active_strategies[strategy_id]
            strategy = strategy_data["strategy"]
            
            # Get current price
            price = market_data.get("close", 0)
            if price <= 0:
                logger.warning(f"Invalid price for {trading_pair}: {price}")
                return
            
            # Determine trade amount
            # In a real implementation, this would use account balance and position sizing
            amount = 0.1  # Example amount
            
            # Execute trade via OKX client
            if signal == 1:  # Buy
                # Check if we already have a position
                if trading_pair in strategy_data["positions"]:
                    logger.info(f"Already have position in {trading_pair}, skipping buy")
                    return
                
                # Execute buy order
                order_result = await okx_client.execute_swap(
                    trading_pair=trading_pair,
                    side="buy",
                    amount=amount,
                    price=price
                )
                
                # Record trade
                trade = {
                    "type": "buy",
                    "trading_pair": trading_pair,
                    "price": price,
                    "amount": amount,
                    "timestamp": time.time(),
                    "order_id": order_result.get("order_id", "unknown")
                }
                strategy_data["trades"].append(trade)
                
                # Update position
                strategy_data["positions"][trading_pair] = {
                    "entry_price": price,
                    "amount": amount,
                    "timestamp": time.time()
                }
                
                logger.info(f"Executed BUY for {trading_pair} at {price}")
                
            elif signal == -1:  # Sell
                # Check if we have a position to sell
                if trading_pair not in strategy_data["positions"]:
                    logger.info(f"No position in {trading_pair}, skipping sell")
                    return
                
                # Get position details
                position = strategy_data["positions"][trading_pair]
                
                # Execute sell order
                order_result = await okx_client.execute_swap(
                    trading_pair=trading_pair,
                    side="sell",
                    amount=position["amount"],
                    price=price
                )
                
                # Record trade
                trade = {
                    "type": "sell",
                    "trading_pair": trading_pair,
                    "price": price,
                    "amount": position["amount"],
                    "timestamp": time.time(),
                    "order_id": order_result.get("order_id", "unknown"),
                    "profit": (price - position["entry_price"]) * position["amount"]
                }
                strategy_data["trades"].append(trade)
                
                # Update performance metrics
                self._update_performance_metrics(strategy_id, trade)
                
                # Remove position
                del strategy_data["positions"][trading_pair]
                
                logger.info(f"Executed SELL for {trading_pair} at {price}")
            
        except Exception as e:
            logger.error(f"Error executing signal for {trading_pair}: {str(e)}")
    
    def _update_performance_metrics(self, strategy_id: str, trade: Dict[str, Any]):
        """Update performance metrics after a trade.
        
        Args:
            strategy_id: ID of the strategy
            trade: Trade details
        """
        try:
            # Get strategy data
            strategy_data = self.active_strategies[strategy_id]
            
            # Only update on sell trades
            if trade["type"] != "sell":
                return
            
            # Update trades count
            strategy_data["performance"]["num_trades"] += 1
            
            # Calculate profit/loss
            profit = trade.get("profit", 0)
            
            # Update total return (simplified)
            strategy_data["performance"]["total_return"] += profit
            
            # Update win rate
            if profit > 0:
                wins = sum(1 for t in strategy_data["trades"] if t["type"] == "sell" and t.get("profit", 0) > 0)
                total_trades = sum(1 for t in strategy_data["trades"] if t["type"] == "sell")
                strategy_data["performance"]["win_rate"] = wins / total_trades if total_trades > 0 else 0
            
            # In a real implementation, calculate more sophisticated metrics
            # like Sharpe ratio and max drawdown
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")


# Create a singleton instance
execution_service = ExecutionService()
