#!/usr/bin/env python
"""
Trading Dashboard Example with WebSockets

This example demonstrates how to create a simple trading dashboard that uses
WebSockets for real-time data streaming. The dashboard displays market data
and trading signals in real-time.

The example includes:
1. A WebSocket server that broadcasts market data and trading signals
2. A simple web-based dashboard that connects to the WebSocket server
3. A market data simulator that generates realistic price movements

Run this example with:
    python examples/trading_dashboard_example.py
"""

import asyncio
import json
import logging
import random
import time
import sys
import os
import datetime
from typing import Dict, Any, List, Optional
import websockets
import pathlib
import webbrowser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Store connected clients
connected_clients = set()

# Store price history for each symbol
price_history = {
    "BTC-USDT": [],
    "ETH-USDT": [],
    "SOL-USDT": []
}

# Base prices for the symbols
base_prices = {
    "BTC-USDT": 50000.0,
    "ETH-USDT": 3000.0,
    "SOL-USDT": 100.0
}

# Current prices
current_prices = base_prices.copy()

# Trading signals
trading_signals = []


class SimpleMovingAverageStrategy:
    """
    A simple moving average crossover trading strategy.
    """
    
    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20):
        """
        Initialize the strategy.
        
        Args:
            symbol: The trading symbol to apply the strategy to
            short_window: The window size for the short-term moving average
            long_window: The window size for the long-term moving average
        """
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.prices: List[float] = []
        self.last_signal: Optional[str] = None
    
    def update(self, price: float) -> Optional[Dict[str, Any]]:
        """
        Update the strategy with a new price and generate a signal if applicable.
        
        Args:
            price: The new price
            
        Returns:
            A trading signal dictionary or None if no signal is generated
        """
        # Add the new price to the price history
        self.prices.append(price)
        
        # We need at least long_window prices to calculate the moving averages
        if len(self.prices) < self.long_window:
            return None
        
        # Calculate the moving averages
        short_ma = sum(self.prices[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices[-self.long_window:]) / self.long_window
        
        # Generate signals based on moving average crossover
        signal = None
        if short_ma > long_ma and (self.last_signal is None or self.last_signal == "SELL"):
            signal = "BUY"
            self.last_signal = signal
        elif short_ma < long_ma and (self.last_signal is None or self.last_signal == "BUY"):
            signal = "SELL"
            self.last_signal = signal
        
        # If we have a signal, return a signal dictionary
        if signal:
            return {
                "symbol": self.symbol,
                "price": price,
                "signal": signal,
                "short_ma": short_ma,
                "long_ma": long_ma,
                "timestamp": time.time()
            }
        
        return None


async def handle_client(websocket):
    """Handle a client connection."""
    # Register client
    connected_clients.add(websocket)
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected")
    
    # Send initial data to the client
    await websocket.send(json.dumps({
        "type": "init",
        "price_history": price_history,
        "trading_signals": trading_signals,
        "timestamp": time.time()
    }))
    
    try:
        # Handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Received from client {client_id}: {data}")
                
                # Handle client messages (e.g., subscribe to specific symbols)
                if data.get("action") == "subscribe":
                    await websocket.send(json.dumps({
                        "type": "subscription",
                        "status": "success",
                        "message": f"Subscribed to all available symbols",
                        "timestamp": time.time()
                    }))
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON from client {client_id}: {message}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": time.time()
                }))
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")
    finally:
        # Unregister client
        connected_clients.remove(websocket)


async def simulate_market_data():
    """Simulate market data and update trading strategies."""
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    strategies = {
        symbol: SimpleMovingAverageStrategy(symbol)
        for symbol in symbols
    }
    
    # Initialize with some historical data
    timestamp = time.time() - 60 * 30  # Start 30 minutes ago
    for _ in range(30):
        for symbol in symbols:
            price_change = current_prices[symbol] * 0.002 * (random.random() * 2 - 1)
            current_prices[symbol] += price_change
            
            # Add to price history
            price_history[symbol].append({
                "price": current_prices[symbol],
                "timestamp": timestamp
            })
            
            # Update strategy
            strategies[symbol].update(current_prices[symbol])
        
        timestamp += 60  # Increment by 1 minute
    
    # Now generate real-time data
    while True:
        current_timestamp = time.time()
        
        for symbol in symbols:
            # Generate random price movement
            price_change = current_prices[symbol] * 0.002 * (random.random() * 2 - 1)
            current_prices[symbol] += price_change
            
            # Add to price history (keep only the last 100 data points)
            price_history[symbol].append({
                "price": current_prices[symbol],
                "timestamp": current_timestamp
            })
            if len(price_history[symbol]) > 100:
                price_history[symbol].pop(0)
            
            # Update strategy and check for signals
            signal_data = strategies[symbol].update(current_prices[symbol])
            if signal_data:
                logger.info(f"Generated {signal_data['signal']} signal for {symbol} at {current_prices[symbol]:.2f}")
                
                # Add to trading signals
                trading_signals.append({
                    "symbol": symbol,
                    "price": current_prices[symbol],
                    "signal": signal_data["signal"],
                    "short_ma": signal_data["short_ma"],
                    "long_ma": signal_data["long_ma"],
                    "timestamp": current_timestamp
                })
                
                # Keep only the last 20 signals
                if len(trading_signals) > 20:
                    trading_signals.pop(0)
            
            # Create market data message
            market_data = {
                "type": "market_data",
                "symbol": symbol,
                "price": current_prices[symbol],
                "volume": random.uniform(10, 100),
                "timestamp": current_timestamp
            }
            
            # Broadcast to all clients
            if connected_clients:
                websockets_tasks = [
                    client.send(json.dumps(market_data))
                    for client in connected_clients
                ]
                
                if websockets_tasks:
                    await asyncio.gather(*websockets_tasks, return_exceptions=True)
        
        # Wait before sending next update
        await asyncio.sleep(1)


def generate_dashboard_html():
    """Generate the HTML for the trading dashboard."""
    html_path = pathlib.Path(__file__).parent / "trading_dashboard.html"
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Trading Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f7fa;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        h1 {
            margin: 0;
        }
        .status {
            background-color: #e74c3c;
            color: white;
            padding: 10px;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .status.connected {
            background-color: #27ae60;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        .card h2 {
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .price {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .price-change {
            font-size: 1.2em;
            margin-left: 10px;
        }
        .price-change.positive {
            color: #27ae60;
        }
        .price-change.negative {
            color: #e74c3c;
        }
        .chart-container {
            height: 200px;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background-color: #f9f9f9;
        }
        .signal {
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 4px;
        }
        .signal.buy {
            background-color: #d5f5e3;
            color: #27ae60;
        }
        .signal.sell {
            background-color: #fce8e8;
            color: #e74c3c;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <header>
        <h1>Crypto Trading Dashboard</h1>
    </header>
    
    <div class="container">
        <div id="connection-status" class="status">Disconnected</div>
        
        <div class="dashboard">
            <div class="card">
                <h2>BTC-USDT</h2>
                <div>
                    <span id="btc-price" class="price">$0.00</span>
                    <span id="btc-change" class="price-change">0.00%</span>
                </div>
                <div class="chart-container">
                    <canvas id="btc-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>ETH-USDT</h2>
                <div>
                    <span id="eth-price" class="price">$0.00</span>
                    <span id="eth-change" class="price-change">0.00%</span>
                </div>
                <div class="chart-container">
                    <canvas id="eth-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>SOL-USDT</h2>
                <div>
                    <span id="sol-price" class="price">$0.00</span>
                    <span id="sol-change" class="price-change">0.00%</span>
                </div>
                <div class="chart-container">
                    <canvas id="sol-chart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Trading Signals</h2>
            <table id="signals-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>Signal</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Signals will be added here -->
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let socket = null;
        let priceHistory = {
            'BTC-USDT': [],
            'ETH-USDT': [],
            'SOL-USDT': []
        };
        let charts = {};
        
        // Initialize charts
        function initCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: false
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            };
            
            charts['BTC-USDT'] = new Chart(
                document.getElementById('btc-chart').getContext('2d'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'BTC-USDT',
                            data: [],
                            borderColor: '#f39c12',
                            tension: 0.1
                        }]
                    },
                    options: chartOptions
                }
            );
            
            charts['ETH-USDT'] = new Chart(
                document.getElementById('eth-chart').getContext('2d'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'ETH-USDT',
                            data: [],
                            borderColor: '#3498db',
                            tension: 0.1
                        }]
                    },
                    options: chartOptions
                }
            );
            
            charts['SOL-USDT'] = new Chart(
                document.getElementById('sol-chart').getContext('2d'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'SOL-USDT',
                            data: [],
                            borderColor: '#9b59b6',
                            tension: 0.1
                        }]
                    },
                    options: chartOptions
                }
            );
        }
        
        // Update price display
        function updatePriceDisplay(symbol, price, change) {
            const symbolPrefix = symbol.split('-')[0].toLowerCase();
            document.getElementById(`${symbolPrefix}-price`).textContent = `$${price.toFixed(2)}`;
            
            const changeElement = document.getElementById(`${symbolPrefix}-change`);
            changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
            
            if (change > 0) {
                changeElement.className = 'price-change positive';
            } else if (change < 0) {
                changeElement.className = 'price-change negative';
            } else {
                changeElement.className = 'price-change';
            }
        }
        
        // Update chart
        function updateChart(symbol, timestamp, price) {
            if (!charts[symbol]) return;
            
            const date = new Date(timestamp * 1000);
            const timeStr = date.toLocaleTimeString();
            
            charts[symbol].data.labels.push(timeStr);
            charts[symbol].data.datasets[0].data.push(price);
            
            // Keep only the last 30 data points
            if (charts[symbol].data.labels.length > 30) {
                charts[symbol].data.labels.shift();
                charts[symbol].data.datasets[0].data.shift();
            }
            
            charts[symbol].update();
        }
        
        // Add trading signal to the table
        function addSignal(signal) {
            const table = document.getElementById('signals-table').getElementsByTagName('tbody')[0];
            const row = table.insertRow(0);
            
            const timeCell = row.insertCell(0);
            const symbolCell = row.insertCell(1);
            const priceCell = row.insertCell(2);
            const signalCell = row.insertCell(3);
            
            const date = new Date(signal.timestamp * 1000);
            timeCell.textContent = date.toLocaleTimeString();
            symbolCell.textContent = signal.symbol;
            priceCell.textContent = `$${signal.price.toFixed(2)}`;
            
            const signalSpan = document.createElement('span');
            signalSpan.textContent = signal.signal;
            signalSpan.className = `signal ${signal.signal.toLowerCase()}`;
            signalCell.appendChild(signalSpan);
            
            // Keep only the last 10 rows
            if (table.rows.length > 10) {
                table.deleteRow(table.rows.length - 1);
            }
        }
        
        // Connect to WebSocket server
        function connect() {
            if (socket) {
                socket.close();
            }
            
            const serverUrl = 'ws://localhost:8765';
            socket = new WebSocket(serverUrl);
            
            socket.onopen = () => {
                console.log('Connected to WebSocket server');
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').className = 'status connected';
                
                // Send subscription message
                socket.send(JSON.stringify({
                    action: 'subscribe'
                }));
            };
            
            socket.onclose = () => {
                console.log('Disconnected from WebSocket server');
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').className = 'status';
                
                // Try to reconnect after 5 seconds
                setTimeout(connect, 5000);
            };
            
            socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'init') {
                        // Initialize with historical data
                        priceHistory = data.price_history;
                        
                        // Initialize charts with historical data
                        for (const symbol in priceHistory) {
                            if (priceHistory[symbol].length > 0) {
                                const prices = priceHistory[symbol];
                                
                                // Calculate price change
                                const firstPrice = prices[0].price;
                                const lastPrice = prices[prices.length - 1].price;
                                const change = ((lastPrice - firstPrice) / firstPrice) * 100;
                                
                                // Update price display
                                updatePriceDisplay(symbol, lastPrice, change);
                                
                                // Update chart
                                for (const dataPoint of prices) {
                                    updateChart(symbol, dataPoint.timestamp, dataPoint.price);
                                }
                            }
                        }
                        
                        // Add trading signals
                        for (const signal of data.trading_signals) {
                            addSignal(signal);
                        }
                    } else if (data.type === 'market_data') {
                        // Update price history
                        if (!priceHistory[data.symbol]) {
                            priceHistory[data.symbol] = [];
                        }
                        
                        priceHistory[data.symbol].push({
                            price: data.price,
                            timestamp: data.timestamp
                        });
                        
                        // Keep only the last 100 data points
                        if (priceHistory[data.symbol].length > 100) {
                            priceHistory[data.symbol].shift();
                        }
                        
                        // Calculate price change (last 24 hours)
                        const prices = priceHistory[data.symbol];
                        if (prices.length > 1) {
                            const lastPrice = prices[prices.length - 1].price;
                            const firstPrice = prices[0].price;
                            const change = ((lastPrice - firstPrice) / firstPrice) * 100;
                            
                            // Update price display
                            updatePriceDisplay(data.symbol, data.price, change);
                            
                            // Update chart
                            updateChart(data.symbol, data.timestamp, data.price);
                        }
                    } else if (data.type === 'signal') {
                        // Add trading signal
                        addSignal(data);
                    }
                } catch (error) {
                    console.error('Error processing message:', error);
                }
            };
        }
        
        // Initialize charts and connect to WebSocket server
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            connect();
        });
    </script>
</body>
</html>
"""
    
    with open(html_path, "w") as f:
        f.write(html_content)
    
    return html_path


async def main():
    """Run the trading dashboard example."""
    # Generate the dashboard HTML
    dashboard_path = generate_dashboard_html()
    
    # Start the WebSocket server
    server = await websockets.serve(handle_client, "localhost", 8765)
    logger.info("WebSocket server started on ws://localhost:8765")
    
    # Open the dashboard in a web browser
    webbrowser.open(f"file://{dashboard_path}")
    
    # Start simulating market data
    market_data_task = asyncio.create_task(simulate_market_data())
    
    # Keep the server running
    await server.wait_closed()


if __name__ == "__main__":
    try:
        print("Starting Trading Dashboard Example...")
        print("This example demonstrates a trading dashboard that uses WebSockets for real-time data streaming.")
        print("The dashboard will open in your web browser automatically.")
        
        # Run the example
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Example stopped by user")
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
