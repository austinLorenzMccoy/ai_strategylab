from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    """Types of trading strategies."""
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MARKET_MAKING = "market_making"
    CUSTOM = "custom"


class RiskTolerance(str, Enum):
    """Risk tolerance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Indicator(BaseModel):
    """Technical indicator definition."""
    name: str
    parameters: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "name": "RSI",
                "parameters": {"period": 14, "overbought": 70, "oversold": 30}
            }
        }


class Condition(BaseModel):
    """Trading condition."""
    indicator: str
    operator: str  # e.g., "<", ">", "==", etc.
    value: Any
    
    class Config:
        schema_extra = {
            "example": {
                "indicator": "RSI",
                "operator": "<",
                "value": 30
            }
        }


class Action(BaseModel):
    """Trading action."""
    type: str  # "BUY", "SELL"
    asset: str
    amount_type: str = "percentage"  # "fixed", "percentage", "all"
    amount: float = 100.0  # Percentage of available balance or fixed amount
    
    class Config:
        schema_extra = {
            "example": {
                "type": "BUY",
                "asset": "ETH",
                "amount_type": "percentage",
                "amount": 50.0
            }
        }


class Rule(BaseModel):
    """Trading rule combining conditions and actions."""
    name: str
    conditions: List[Condition]
    actions: List[Action]
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Buy ETH when RSI is low",
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
            }
        }


class Strategy(BaseModel):
    """Complete trading strategy definition."""
    id: Optional[str] = None
    name: str
    description: str
    type: StrategyType
    trading_pairs: List[str]
    risk_tolerance: RiskTolerance = RiskTolerance.MEDIUM
    indicators: List[Indicator]
    rules: List[Rule]
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "RSI Oversold Bounce",
                "description": "Buy when RSI is oversold and sell when overbought",
                "type": "mean_reversion",
                "trading_pairs": ["ETH-USDT"],
                "risk_tolerance": "medium",
                "indicators": [
                    {
                        "name": "RSI",
                        "parameters": {"period": 14, "overbought": 70, "oversold": 30}
                    }
                ],
                "rules": [
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
                "parameters": {
                    "stop_loss": 5.0,
                    "take_profit": 10.0
                },
                "metadata": {
                    "created_at": "2025-05-16T10:00:00Z",
                    "created_by": "AI Strategy Lab"
                }
            }
        }


class StrategyRequest(BaseModel):
    """Request model for generating a strategy."""
    type: StrategyType
    trading_pairs: List[str]
    risk_tolerance: RiskTolerance = RiskTolerance.MEDIUM
    description: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "type": "mean_reversion",
                "trading_pairs": ["ETH-USDT"],
                "risk_tolerance": "medium",
                "description": "Create a strategy that buys ETH when it's oversold",
                "constraints": {
                    "max_position_size": 0.5,
                    "min_profit_target": 2.0
                }
            }
        }


class BacktestResult(BaseModel):
    """Result of strategy backtesting."""
    strategy_id: str
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: int
    detailed_metrics: Optional[Dict[str, Any]] = None


class StrategyPerformance(BaseModel):
    """Strategy performance metrics."""
    strategy_id: str
    start_time: str
    current_time: str
    total_return: float
    current_positions: Dict[str, Any]
    recent_trades: List[Dict[str, Any]]
    metrics: Dict[str, Any]