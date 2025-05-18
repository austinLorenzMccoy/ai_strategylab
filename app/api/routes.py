from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from typing import List, Dict, Any, Optional
import uuid
import logging
from app.models.strategy import Strategy, StrategyRequest, StrategyType, RiskTolerance, BacktestResult, StrategyPerformance
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service
from app.services.rl_service import rl_service
from app.services.execution_service import execution_service
from app.core.okx_client import okx_client
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/strategies", response_model=List[Strategy])
async def get_strategies():
    """Get all available strategies."""
    try:
        # In a real application, this would fetch from a database
        # For now, we'll return a sample strategy
        sample_strategy = Strategy(
            id="sample-strategy-1",
            name="Sample RSI Strategy",
            description="Buy when RSI is oversold and sell when overbought",
            type=StrategyType.MEAN_REVERSION,
            trading_pairs=["ETH-USDT"],
            risk_tolerance=RiskTolerance.MEDIUM,
            indicators=[
                {
                    "name": "RSI",
                    "parameters": {"period": 14, "overbought": 70, "oversold": 30}
                }
            ],
            rules=[
                {
                    "name": "Buy when oversold",
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
                            "asset": "ETH",
                            "amount_type": "percentage",
                            "amount": 50.0
                        }
                    ]
                },
                {
                    "name": "Sell when overbought",
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
                            "asset": "ETH",
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
        return [sample_strategy]
    except Exception as e:
        logger.error(f"Error fetching strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategies: {str(e)}")


@router.get("/strategies/{strategy_id}", response_model=Strategy)
async def get_strategy(strategy_id: str = Path(..., description="The ID of the strategy to retrieve")):
    """Get a specific strategy by ID."""
    try:
        # In a real application, this would fetch from a database
        # For now, we'll return a sample strategy if the ID matches
        if strategy_id == "sample-strategy-1":
            return Strategy(
                id="sample-strategy-1",
                name="Sample RSI Strategy",
                description="Buy when RSI is oversold and sell when overbought",
                type=StrategyType.MEAN_REVERSION,
                trading_pairs=["ETH-USDT"],
                risk_tolerance=RiskTolerance.MEDIUM,
                indicators=[
                    {
                        "name": "RSI",
                        "parameters": {"period": 14, "overbought": 70, "oversold": 30}
                    }
                ],
                rules=[
                    {
                        "name": "Buy when oversold",
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
                                "asset": "ETH",
                                "amount_type": "percentage",
                                "amount": 50.0
                            }
                        ]
                    },
                    {
                        "name": "Sell when overbought",
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
                                "asset": "ETH",
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
        else:
            raise HTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategy: {str(e)}")


@router.post("/strategies", response_model=Strategy)
async def create_strategy(strategy_request: StrategyRequest, background_tasks: BackgroundTasks):
    """Generate a new trading strategy."""
    try:
        # Generate a strategy using our services
        strategy_id = f"strategy-{uuid.uuid4()}"
        
        # This would typically be an async operation that uses our RAG and LLM services
        # For now, we'll create a simple strategy based on the request
        strategy = Strategy(
            id=strategy_id,
            name=f"{strategy_request.type.capitalize()} Strategy for {', '.join(strategy_request.trading_pairs)}",
            description=strategy_request.description or f"AI-generated {strategy_request.type} strategy",
            type=strategy_request.type,
            trading_pairs=strategy_request.trading_pairs,
            risk_tolerance=strategy_request.risk_tolerance,
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
                            "asset": strategy_request.trading_pairs[0].split('-')[0],
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
                            "asset": strategy_request.trading_pairs[0].split('-')[0],
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
        
        # In a real application, we would optimize the strategy using RL
        # and store it in a database
        background_tasks.add_task(rl_service.optimize_strategy, strategy_id)
        
        return strategy
    
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create strategy: {str(e)}")


@router.post("/strategies/{strategy_id}/backtest", response_model=BacktestResult)
async def backtest_strategy(
    strategy_id: str = Path(..., description="The ID of the strategy to backtest"),
    start_date: str = Query(..., description="Start date for backtesting (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date for backtesting (YYYY-MM-DD)")
):
    """Backtest a strategy against historical data."""
    try:
        # In a real application, this would fetch the strategy and run a backtest
        # For now, we'll return sample backtest results
        return BacktestResult(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date,
            total_return=15.7,
            sharpe_ratio=1.8,
            max_drawdown=-8.5,
            win_rate=0.65,
            trades=42,
            detailed_metrics={
                "monthly_returns": [2.1, 3.5, -1.2, 4.5, 6.8],
                "avg_trade_duration": "2.3 days",
                "profit_factor": 2.1
            }
        )
    except Exception as e:
        logger.error(f"Error backtesting strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to backtest strategy: {str(e)}")


@router.post("/strategies/{strategy_id}/deploy", response_model=Dict[str, Any])
async def deploy_strategy(strategy_id: str = Path(..., description="The ID of the strategy to deploy")):
    """Deploy a strategy for live trading."""
    try:
        # In a real application, this would deploy the strategy for execution
        # For now, we'll return a success message
        return {
            "status": "success",
            "message": f"Strategy {strategy_id} deployed successfully",
            "deployment_id": f"deployment-{uuid.uuid4()}",
            "timestamp": "2025-05-16T12:34:56Z"
        }
    except Exception as e:
        logger.error(f"Error deploying strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy strategy: {str(e)}")


@router.get("/strategies/{strategy_id}/performance", response_model=StrategyPerformance)
async def get_strategy_performance(strategy_id: str = Path(..., description="The ID of the strategy to check")):
    """Get performance metrics for a deployed strategy."""
    try:
        # In a real application, this would fetch live performance data
        # For now, we'll return sample performance data
        return StrategyPerformance(
            strategy_id=strategy_id,
            start_time="2025-05-10T00:00:00Z",
            current_time="2025-05-16T12:34:56Z",
            total_return=8.3,
            current_positions={
                "ETH": {
                    "amount": 0.5,
                    "entry_price": 5200.0,
                    "current_price": 5350.0,
                    "profit_loss": 2.9
                }
            },
            recent_trades=[
                {
                    "timestamp": "2025-05-15T14:23:45Z",
                    "action": "BUY",
                    "asset": "ETH",
                    "amount": 0.5,
                    "price": 5200.0
                },
                {
                    "timestamp": "2025-05-14T09:12:34Z",
                    "action": "SELL",
                    "asset": "ETH",
                    "amount": 0.3,
                    "price": 5150.0
                }
            ],
            metrics={
                "sharpe_ratio": 1.6,
                "max_drawdown": -3.2,
                "win_rate": 0.7,
                "trades": 12
            }
        )
    except Exception as e:
        logger.error(f"Error fetching performance for strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategy performance: {str(e)}")


@router.get("/market/data/{trading_pair}", response_model=Dict[str, Any])
async def get_market_data(trading_pair: str = Path(..., description="Trading pair (e.g., BTC-USDT)")):
    """Get current market data for a trading pair."""
    try:
        # Call OKX API client to get market data
        market_data = await okx_client.get_market_data(trading_pair)
        return market_data
    except Exception as e:
        logger.error(f"Error fetching market data for {trading_pair}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")


@router.get("/market/klines/{trading_pair}", response_model=Dict[str, Any])
async def get_kline_data(
    trading_pair: str = Path(..., description="Trading pair (e.g., BTC-USDT)"),
    bar: str = Query("1m", description="Candlestick interval (e.g., 1m, 5m, 1h)"),
    limit: int = Query(100, description="Number of candlesticks to return")
):
    """Get candlestick (kline) data for a trading pair."""
    try:
        # Call OKX API client to get kline data
        kline_data = await okx_client.get_kline_data(trading_pair, bar, limit)
        return kline_data
    except Exception as e:
        logger.error(f"Error fetching kline data for {trading_pair}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch kline data: {str(e)}")