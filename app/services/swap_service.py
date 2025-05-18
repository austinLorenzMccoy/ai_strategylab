"""
Swap Service Module

This module provides a service for interacting with the OKX Swap API,
enabling the execution of cryptocurrency swaps with function calling capabilities.
It integrates with the WebSocket system for real-time updates and notifications.

Key features:
- Execute swaps between cryptocurrencies
- Simulate transactions to estimate impact
- Monitor swap execution status
- Provide real-time updates via WebSockets
- Function calling interface for agent integration

This implementation allows for seamless integration with the multi-agent system,
enabling agents to execute swaps based on their decision-making processes.
"""

import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

from app.core.okx_client import okx_client
from app.services.websocket_service import MessagePublisher
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SwapRequest(BaseModel):
    """Model for a swap request."""
    from_currency: str
    to_currency: str
    amount: float
    side: str = "buy"  # "buy" or "sell"
    client_order_id: Optional[str] = None
    price_limit: Optional[float] = None  # Limit price for the swap
    strategy_id: Optional[str] = None  # ID of the strategy initiating the swap
    agent_id: Optional[str] = None  # ID of the agent initiating the swap


class SwapResult(BaseModel):
    """Model for a swap result."""
    request_id: str
    from_currency: str
    to_currency: str
    amount: float
    executed_price: float
    executed_amount: float
    fee: float
    timestamp: str
    status: str  # "success", "failed", "pending"
    error_message: Optional[str] = None
    transaction_id: Optional[str] = None


class SwapService:
    """Service for executing cryptocurrency swaps via OKX API."""
    
    def __init__(self, websocket_channel: Optional[str] = None):
        """Initialize the swap service.
        
        Args:
            websocket_channel: Optional channel for publishing swap updates
        """
        self.websocket_channel = websocket_channel
        self.pending_swaps = {}  # Track pending swap requests
        self.swap_history = []  # Store swap history
    
    async def execute_swap(self, request: SwapRequest) -> Dict[str, Any]:
        """Execute a swap between two cryptocurrencies.
        
        Args:
            request: Swap request parameters
            
        Returns:
            Swap execution result
        """
        try:
            logger.info(f"Executing swap: {request.from_currency} to {request.to_currency}, amount: {request.amount}")
            
            # Generate a request ID if not provided
            request_id = request.client_order_id or f"swap_{int(time.time() * 1000)}"
            
            # Store the request in pending swaps
            self.pending_swaps[request_id] = {
                "request": request.dict(),
                "status": "pending",
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish the pending status
            if self.websocket_channel:
                await self._publish_swap_status(request_id, "pending")
            
            # Execute the swap via OKX API
            result = await okx_client.execute_swap(
                from_ccy=request.from_currency,
                to_ccy=request.to_currency,
                amount=request.amount,
                side=request.side
            )
            
            # Check for errors
            if "error" in result:
                logger.error(f"Swap execution failed: {result['error']}")
                
                # Update pending swap status
                self.pending_swaps[request_id]["status"] = "failed"
                self.pending_swaps[request_id]["error"] = result["error"]
                
                # Publish the failed status
                if self.websocket_channel:
                    await self._publish_swap_status(request_id, "failed", error=result["error"])
                
                return {
                    "status": "failed",
                    "request_id": request_id,
                    "error": result["error"]
                }
            
            # Extract swap details from result
            swap_details = result.get("data", {})
            
            # Create swap result
            swap_result = SwapResult(
                request_id=request_id,
                from_currency=request.from_currency,
                to_currency=request.to_currency,
                amount=request.amount,
                executed_price=float(swap_details.get("fillPx", 0)),
                executed_amount=float(swap_details.get("fillSz", 0)),
                fee=float(swap_details.get("fee", 0)),
                timestamp=swap_details.get("cTime", datetime.now().isoformat()),
                status="success",
                transaction_id=swap_details.get("ordId", "")
            )
            
            # Update pending swap status
            self.pending_swaps[request_id]["status"] = "success"
            self.pending_swaps[request_id]["result"] = swap_result.dict()
            
            # Add to swap history
            self.swap_history.append({
                "request_id": request_id,
                "request": request.dict(),
                "result": swap_result.dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Publish the success status
            if self.websocket_channel:
                await self._publish_swap_status(request_id, "success", result=swap_result.dict())
            
            return {
                "status": "success",
                "request_id": request_id,
                "result": swap_result.dict()
            }
            
        except Exception as e:
            logger.error(f"Error executing swap: {str(e)}")
            
            # Update pending swap status if request_id exists
            if request.client_order_id and request.client_order_id in self.pending_swaps:
                request_id = request.client_order_id
                self.pending_swaps[request_id]["status"] = "failed"
                self.pending_swaps[request_id]["error"] = str(e)
                
                # Publish the failed status
                if self.websocket_channel:
                    await self._publish_swap_status(request_id, "failed", error=str(e))
            
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def simulate_swap(self, request: SwapRequest) -> Dict[str, Any]:
        """Simulate a swap to estimate impact without executing it.
        
        Args:
            request: Swap request parameters
            
        Returns:
            Simulation result
        """
        try:
            logger.info(f"Simulating swap: {request.from_currency} to {request.to_currency}, amount: {request.amount}")
            
            # Simulate the swap via OKX API
            result = await okx_client.simulate_transaction(
                from_ccy=request.from_currency,
                to_ccy=request.to_currency,
                amount=request.amount,
                side=request.side
            )
            
            # Check for errors
            if "error" in result:
                logger.error(f"Swap simulation failed: {result['error']}")
                return {
                    "status": "failed",
                    "error": result["error"]
                }
            
            # Extract simulation details
            simulation = result.get("data", {})
            
            # Return simulation result
            return {
                "status": "success",
                "from_currency": request.from_currency,
                "to_currency": request.to_currency,
                "amount": request.amount,
                "estimated_price": float(simulation.get("estimatedPx", 0)),
                "estimated_amount": float(simulation.get("estimatedSz", 0)),
                "estimated_fee": float(simulation.get("estimatedFee", 0)),
                "price_impact": float(simulation.get("priceImpact", 0)) * 100,  # Convert to percentage
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error simulating swap: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_swap_status(self, request_id: str) -> Dict[str, Any]:
        """Get the status of a swap request.
        
        Args:
            request_id: ID of the swap request
            
        Returns:
            Swap status
        """
        if request_id not in self.pending_swaps:
            return {
                "status": "not_found",
                "message": f"Swap request {request_id} not found"
            }
        
        return {
            "status": "found",
            "swap_status": self.pending_swaps[request_id]["status"],
            "details": self.pending_swaps[request_id]
        }
    
    async def get_swap_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get swap execution history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            Swap history
        """
        return self.swap_history[-limit:]
    
    async def _publish_swap_status(self, request_id: str, status: str, error: str = None, result: Dict[str, Any] = None):
        """Publish swap status to WebSocket channel.
        
        Args:
            request_id: ID of the swap request
            status: Status of the swap
            error: Optional error message
            result: Optional swap result
        """
        if not self.websocket_channel:
            return
        
        message = {
            "type": "swap_status",
            "request_id": request_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            message["error"] = error
        
        if result:
            message["result"] = result
        
        await MessagePublisher.publish(self.websocket_channel, message)


# Create a singleton instance
swap_service = SwapService(websocket_channel="swap_updates")
