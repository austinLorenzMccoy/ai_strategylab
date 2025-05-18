import os
from functools import lru_cache
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables."""
    
    # App settings
    APP_NAME: str = os.getenv("APP_NAME", "AI_Strategy_Lab")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ["true", "1", "t"]
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # FastAPI settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # OKX API settings
    OKX_API_KEY: str = os.getenv("OKX_API_KEY", "")
    OKX_SECRET_KEY: str = os.getenv("OKX_SECRET_KEY", "")
    OKX_PASSPHRASE: str = os.getenv("OKX_PASSPHRASE", "")
    OKX_API_BASE_URL: str = os.getenv("OKX_API_BASE_URL", "https://www.okx.com")
    
    # LLM settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    
    # RAG settings
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # RL settings
    RL_MODEL_PATH: str = os.getenv("RL_MODEL_PATH", "./models/rl")
    SIMULATION_ENVIRONMENT: str = os.getenv("SIMULATION_ENVIRONMENT", "OKXMarketEnv")
    
    # Strategy settings
    DEFAULT_RISK_TOLERANCE: str = os.getenv("DEFAULT_RISK_TOLERANCE", "medium")
    DEFAULT_TRADING_PAIRS: List[str] = os.getenv("DEFAULT_TRADING_PAIRS", "BTC-USDT,ETH-USDT").split(",")
    STRATEGY_TEMPLATES_PATH: str = os.getenv("STRATEGY_TEMPLATES_PATH", "./data/strategies")
    
    # Notification settings (optional)
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    
    class Config:
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Create and cache settings instance."""
    return Settings()