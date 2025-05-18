#!/usr/bin/env python
"""
OKX Trading Dashboard Example with WebSockets

This example demonstrates how to create a trading dashboard that uses
WebSockets for real-time data streaming with actual OKX market data.
The dashboard displays real market data and trading signals in real-time.

The example includes:
1. A WebSocket server that broadcasts market data and trading signals
2. A web-based dashboard that connects to the WebSocket server
3. Integration with the OKX API for real market data

Run this example with:
    python examples/trading_dashboard_okx.py
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
import aiohttp
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OKX API credentials
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")
OKX_API_BASE_URL = os.getenv("OKX_API_BASE_URL", "https://www.okx.com")

# Store connected clients
connected_clients = set()

# Store price history for each symbol
price_history = {
    "BTC-USDT": [],
    "ETH-USDT": [],
    "SOL-USDT": []
}

# Current prices
current_prices = {
    "BTC-USDT": 0.0,
    "ETH-USDT": 0.0,
    "SOL-USDT": 0.0
}

# Trading signals
trading_signals = []


class SimpleMovingAverageStrategy:
    """A simple moving average crossover trading strategy."""
    
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
        self.prices.append(price)
        
        # Keep only the necessary price history
        if len(self.prices) > self.long_window:
            self.prices = self.prices[-self.long_window:]
        
        # Need enough prices to calculate both moving averages
        if len(self.prices) < self.long_window:
            return None
        
        # Calculate moving averages
        short_ma = sum(self.prices[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices) / len(self.prices)
        
        # Generate signals based on moving average crossover
        signal = None
        if short_ma > long_ma and (self.last_signal is None or self.last_signal == "SELL"):
            signal = "BUY"
            self.last_signal = signal
        elif short_ma < long_ma and (self.last_signal is None or self.last_signal == "BUY"):
            signal = "SELL"
            self.last_signal = signal
        
        if signal:
            return {
                "symbol": self.symbol,
                "price": price,
                "signal": signal,
                "timestamp": datetime.datetime.now().isoformat(),
                "short_ma": short_ma,
                "long_ma": long_ma
            }
        
        return None


async def handle_client(websocket):
    """Handle a client connection."""
    client_id = id(websocket)
    connected_clients.add(websocket)
    logger.info(f"Client {client_id} connected")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Received from client {client_id}: {data}")
                
                # Handle client messages
                if data.get("action") == "subscribe":
                    # Send initial data
                    for symbol in price_history:
                        if price_history[symbol]:
                            await websocket.send(json.dumps({
                                "type": "price_history",
                                "symbol": symbol,
                                "data": price_history[symbol]
                            }))
                    
                    # Send trading signals
                    if trading_signals:
                        await websocket.send(json.dumps({
                            "type": "trading_signals",
                            "data": trading_signals
                        }))
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from client {client_id}: {message}")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")
    finally:
        connected_clients.remove(websocket)


async def fetch_okx_market_data():
    """Fetch market data from OKX API."""
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    
    # OKX uses a different format for instrument IDs
    # For spot trading, the format is "BTC-USDT-SPOT"
    okx_symbols = [f"{s.split('-')[0]}-{s.split('-')[1]}-SPOT" for s in symbols]
    
    # Log API credentials (without sensitive info)
    logger.info(f"OKX API Base URL: {OKX_API_BASE_URL}")
    logger.info(f"OKX API Key available: {bool(OKX_API_KEY)}")
    logger.info(f"Using OKX symbols: {okx_symbols}")
    
    strategies = {
        symbol: SimpleMovingAverageStrategy(symbol)
        for symbol in symbols
    }
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                for i, symbol in enumerate(symbols):
                    okx_symbol = okx_symbols[i]
                    
                    # Fetch ticker data from OKX
                    url = f"{OKX_API_BASE_URL}/api/v5/market/ticker?instId={okx_symbol}"
                    logger.info(f"Fetching data from: {url}")
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Response for {symbol}: {json.dumps(data)}")
                            
                            if data.get("code") == "0" and data.get("data"):
                                ticker = data["data"][0]
                                price = float(ticker["last"])
                                logger.info(f"Extracted price for {symbol}: {price}")
                                
                                # Update current price
                                current_prices[symbol] = price
                                
                                # Add to price history
                                timestamp = time.time()
                                price_history[symbol].append({
                                    "price": price,
                                    "timestamp": timestamp
                                })
                                
                                # Keep only the last 100 price points
                                if len(price_history[symbol]) > 100:
                                    price_history[symbol] = price_history[symbol][-100:]
                                
                                # Update strategy and check for signals
                                signal = strategies[symbol].update(price)
                                if signal:
                                    logger.info(f"Generated {signal['signal']} signal for {symbol} at {price:.2f}")
                                    trading_signals.append(signal)
                                    
                                    # Keep only the last 20 signals
                                    if len(trading_signals) > 20:
                                        trading_signals.pop(0)
                                    
                                    # Broadcast the signal to all clients
                                    for client in connected_clients:
                                        try:
                                            await client.send(json.dumps({
                                                "type": "trading_signal",
                                                "data": signal
                                            }))
                                        except Exception as e:
                                            logger.error(f"Error sending signal to client: {str(e)}")
                                
                                # Broadcast the price update to all clients
                                logger.info(f"Broadcasting price update for {symbol}: {price}")
                                for client in connected_clients:
                                    try:
                                        await client.send(json.dumps({
                                            "type": "price_update",
                                            "symbol": symbol,
                                            "data": {
                                                "price": price,
                                                "timestamp": timestamp
                                            }
                                        }))
                                        logger.info(f"Sent price update to client for {symbol}")
                                    except Exception as e:
                                        logger.error(f"Error sending price update to client: {str(e)}")
                            else:
                                logger.error(f"Invalid data format for {symbol}: {data}")
                        else:
                            response_text = await response.text()
                            logger.error(f"Error fetching data for {symbol}: {response.status}, Response: {response_text}")
                
                # Wait before the next update
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in market data task: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error


def generate_dashboard_html():
    """Generate the HTML for the trading dashboard."""
    # Create the examples directory if it doesn't exist
    examples_dir = pathlib.Path(__file__).parent
    examples_dir.mkdir(exist_ok=True)
    
    # Path for the dashboard HTML file
    html_path = examples_dir / "okx_dashboard.html"
    
    # Generate the HTML content
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OKX Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #2a2a2a;
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
        }
        .connection-status {
            display: flex;
            align-items: center;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .connected {
            background-color: #4CAF50;
        }
        .disconnected {
            background-color: #F44336;
        }
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin: 0;
        }
        .price-info {
            text-align: right;
        }
        .current-price {
            font-size: 20px;
            font-weight: bold;
            margin: 0;
        }
        .price-change {
            margin: 5px 0 0 0;
            font-size: 14px;
        }
        .positive {
            color: #4CAF50;
        }
        .negative {
            color: #F44336;
        }
        .signals-container {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .signals-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .signals-title {
            font-size: 18px;
            font-weight: bold;
            margin: 0;
        }
        .signals-table {
            width: 100%;
            border-collapse: collapse;
        }
        .signals-table th, .signals-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .signals-table th {
            background-color: #f2f2f2;
        }
        .buy-signal {
            color: #4CAF50;
            font-weight: bold;
        }
        .sell-signal {
            color: #F44336;
            font-weight: bold;
        }
        .chart-container {
            position: relative;
            height: 250px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OKX Trading Dashboard</h1>
            <div class="connection-status">
                <div id="status-indicator" class="status-indicator disconnected"></div>
                <span id="connection-status">Disconnected</span>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart-card">
                <div class="chart-header">
                    <h2 class="chart-title">BTC-USDT</h2>
                    <div class="price-info">
                        <p id="btc-price" class="current-price">$0.00</p>
                        <p id="btc-change" class="price-change">0.00%</p>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="btc-chart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <h2 class="chart-title">ETH-USDT</h2>
                    <div class="price-info">
                        <p id="eth-price" class="current-price">$0.00</p>
                        <p id="eth-change" class="price-change">0.00%</p>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="eth-chart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <h2 class="chart-title">SOL-USDT</h2>
                    <div class="price-info">
                        <p id="sol-price" class="current-price">$0.00</p>
                        <p id="sol-change" class="price-change">0.00%</p>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="sol-chart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="signals-container">
            <div class="signals-header">
                <h2 class="signals-title">Trading Signals</h2>
            </div>
            <table class="signals-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>Signal</th>
                    </tr>
                </thead>
                <tbody id="signals-body">
                    <!-- Signals will be inserted here -->
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let socket;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        const reconnectDelay = 3000; // 3 seconds
        
        // Chart objects
        const charts = {};
        const priceData = {
            'BTC-USDT': [],
            'ETH-USDT': [],
            'SOL-USDT': []
        };
        
        // Previous prices for calculating change
        const previousPrices = {
            'BTC-USDT': 0,
            'ETH-USDT': 0,
            'SOL-USDT': 0
        };
        
        // Initialize charts
        function initCharts() {
            const symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT'];
            const colors = {
                'BTC-USDT': 'rgb(247, 147, 26)',
                'ETH-USDT': 'rgb(98, 126, 234)',
                'SOL-USDT': 'rgb(20, 241, 149)'
            };
            
            symbols.forEach(symbol => {
                const shortSymbol = symbol.split('-')[0].toLowerCase();
                const ctx = document.getElementById(`${shortSymbol}-chart`).getContext('2d');
                
                charts[symbol] = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: symbol,
                            data: [],
                            borderColor: colors[symbol],
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.1,
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                display: true,
                                title: {
                                    display: false
                                },
                                ticks: {
                                    maxTicksLimit: 5
                                }
                            },
                            y: {
                                display: true,
                                title: {
                                    display: false
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false
                            }
                        },
                        animation: {
                            duration: 0
                        }
                    }
                });
            });
        }
        
        // Update chart with new price data
        function updateChart(symbol, price, timestamp) {
            if (!charts[symbol]) return;
            
            const time = new Date(timestamp * 1000).toLocaleTimeString();
            const chart = charts[symbol];
            
            // Add new data point
            chart.data.labels.push(time);
            chart.data.datasets[0].data.push(price);
            
            // Keep only the last 30 data points
            if (chart.data.labels.length > 30) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            // Update the chart
            chart.update();
            
            // Update price display
            const shortSymbol = symbol.split('-')[0].toLowerCase();
            const priceElement = document.getElementById(`${shortSymbol}-price`);
            const changeElement = document.getElementById(`${shortSymbol}-change`);
            
            priceElement.textContent = `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            
            // Calculate and display price change
            if (previousPrices[symbol] > 0) {
                const change = ((price - previousPrices[symbol]) / previousPrices[symbol]) * 100;
                const changeText = change.toFixed(2) + '%';
                
                changeElement.textContent = changeText;
                if (change >= 0) {
                    changeElement.className = 'price-change positive';
                    changeElement.textContent = '+' + changeText;
                } else {
                    changeElement.className = 'price-change negative';
                }
            }
            
            // Update previous price
            previousPrices[symbol] = price;
        }
        
        // Add a new trading signal to the table
        function addSignal(signal) {
            const signalsBody = document.getElementById('signals-body');
            const row = document.createElement('tr');
            
            // Format the timestamp
            const time = new Date(signal.timestamp).toLocaleTimeString();
            
            // Create the signal row
            row.innerHTML = `
                <td>${time}</td>
                <td>${signal.symbol}</td>
                <td>$${signal.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                <td class="${signal.signal.toLowerCase()}-signal">${signal.signal}</td>
            `;
            
            // Add the row to the table
            signalsBody.prepend(row);
            
            // Keep only the last 10 signals
            while (signalsBody.children.length > 10) {
                signalsBody.removeChild(signalsBody.lastChild);
            }
        }
        
        // Connect to the WebSocket server
        function connectWebSocket() {
            // Update connection status
            const statusIndicator = document.getElementById('status-indicator');
            const connectionStatus = document.getElementById('connection-status');
            statusIndicator.className = 'status-indicator disconnected';
            connectionStatus.textContent = 'Connecting...';
            
            // Create WebSocket connection
            socket = new WebSocket('ws://localhost:8765');
            
            // Connection opened
            socket.addEventListener('open', (event) => {
                console.log('Connected to WebSocket server');
                statusIndicator.className = 'status-indicator connected';
                connectionStatus.textContent = 'Connected';
                reconnectAttempts = 0;
                
                // Subscribe to updates
                socket.send(JSON.stringify({ action: 'subscribe' }));
            });
            
            // Listen for messages
            socket.addEventListener('message', (event) => {
                try {
                    const message = JSON.parse(event.data);
                    
                    if (message.type === 'price_update') {
                        // Update chart with new price data
                        updateChart(message.symbol, message.data.price, message.data.timestamp);
                    } else if (message.type === 'price_history') {
                        // Initialize chart with historical data
                        message.data.forEach(item => {
                            updateChart(message.symbol, item.price, item.timestamp);
                        });
                    } else if (message.type === 'trading_signal') {
                        // Add trading signal
                        addSignal(message.data);
                    } else if (message.type === 'trading_signals') {
                        // Add multiple trading signals
                        message.data.forEach(signal => {
                            addSignal(signal);
                        });
                    }
                } catch (error) {
                    console.error('Error processing message:', error);
                }
            });
            
            // Connection closed
            socket.addEventListener('close', (event) => {
                console.log('Disconnected from WebSocket server');
                statusIndicator.className = 'status-indicator disconnected';
                connectionStatus.textContent = 'Disconnected';
                
                // Attempt to reconnect
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    console.log(`Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`);
                    setTimeout(connectWebSocket, reconnectDelay);
                } else {
                    console.log('Max reconnect attempts reached');
                    connectionStatus.textContent = 'Disconnected (Max attempts reached)';
                }
            });
            
            // Connection error
            socket.addEventListener('error', (event) => {
                console.error('WebSocket error:', event);
            });
        }
        
        // Initialize the dashboard
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            connectWebSocket();
        });
    </script>
</body>
</html>
    """
    
    # Write the HTML content to the file
    with open(html_path, "w") as f:
        f.write(html_content)
    
    return html_path


async def main():
    """Run the OKX trading dashboard example."""
    print("Starting OKX Trading Dashboard Example...")
    print("This example demonstrates a trading dashboard that uses WebSockets for real-time OKX market data.")
    print("The dashboard will open in your web browser automatically.")
    
    # Generate the dashboard HTML
    dashboard_path = generate_dashboard_html()
    
    # Start the WebSocket server
    server = await websockets.serve(handle_client, "localhost", 8765)
    logger.info("WebSocket server started on ws://localhost:8765")
    
    # Open the dashboard in a web browser
    webbrowser.open(f"file://{dashboard_path}")
    
    # Start fetching market data
    market_data_task = asyncio.create_task(fetch_okx_market_data())
    
    # Keep the server running
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting OKX Trading Dashboard Example...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Exiting OKX Trading Dashboard Example...")
    finally:
        print("Goodbye!")
        sys.exit(0)
