import time
import hmac
import base64
import json
import logging
from typing import Dict, Any, Optional, List
import httpx
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OKXClient:
    """Client for interacting with OKX APIs."""
    
    def __init__(self):
        self.api_key = settings.OKX_API_KEY
        self.api_secret = settings.OKX_SECRET_KEY
        self.passphrase = settings.OKX_PASSPHRASE
        self.base_url = settings.OKX_API_BASE_URL
        self.timeout = 10.0  # seconds
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate OKX API signature."""
        if not body:
            body = ""
            
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.api_secret, encoding="utf-8"),
            bytes(message, encoding="utf-8"),
            digestmod="sha256",
        )
        return base64.b64encode(mac.digest()).decode()
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an authenticated request to OKX API."""
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time()))
        
        # Prepare request body
        body = ""
        if data:
            body = json.dumps(data)
        
        # Generate signature
        signature = self._generate_signature(timestamp, method.upper(), endpoint, body)
        
        # Prepare headers
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method.lower() == "get":
                    response = await client.get(url, headers=headers, params=params)
                elif method.lower() == "post":
                    response = await client.post(url, headers=headers, json=data)
                else:
                    logger.error(f"Unsupported HTTP method: {method}")
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error making request: {e}")
            return {"error": str(e)}
    
    # Market Data API methods
    async def get_market_data(self, trading_pair: str) -> Dict[str, Any]:
        """Get market data for a trading pair."""
        endpoint = f"/api/v5/market/ticker?instId={trading_pair}"
        return await self._request("get", endpoint)
    
    async def get_kline_data(self, trading_pair: str, bar: str = "1m", limit: int = 100) -> Dict[str, Any]:
        """Get candlestick data."""
        endpoint = f"/api/v5/market/candles?instId={trading_pair}&bar={bar}&limit={limit}"
        return await self._request("get", endpoint)
    
    # Swap API methods
    async def execute_swap(self, from_ccy: str, to_ccy: str, amount: float) -> Dict[str, Any]:
        """Execute a swap between two currencies."""
        endpoint = "/api/v5/trade/swap"
        data = {
            "fromCcy": from_ccy,
            "toCcy": to_ccy,
            "amount": str(amount),
            "side": "buy"  # or "sell"
        }
        return await self._request("post", endpoint, data=data)
    
    # Transaction Simulation API
    async def simulate_transaction(self, from_ccy: str, to_ccy: str, amount: float) -> Dict[str, Any]:
        """Simulate a transaction to check impact."""
        endpoint = "/api/v5/trade/estimate-quote"
        data = {
            "fromCcy": from_ccy,
            "toCcy": to_ccy,
            "amount": str(amount),
            "side": "buy"  # or "sell"
        }
        return await self._request("post", endpoint, data=data)
    
    # WebSocket connections for real-time data can be implemented here
    # This would require a separate WebSocket client implementation
    # For simplicity, we're not including it in this basic version


# Create a singleton instance
okx_client = OKXClient()