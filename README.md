# Trading Signal Dashboard

This project implements a simple web‑based dashboard that monitors one or more trading instruments using the Interactive Brokers (IBKR) API and highlights when a potential buy or sell opportunity is approaching.  It combines a Python/FastAPI backend with a lightweight HTML/JavaScript frontend.  The backend fetches real‑time market data via the IBKR TWS/Gateway API, computes a rich set of technical indicators (RSI, MACD, moving averages, Bollinger Bands, ATR, Stochastic and more) and identifies candlestick patterns.  It exposes these metrics over REST.  The frontend polls these endpoints periodically, displays the latest values and visually indicates how close each instrument is to a trading signal.

> **Important**: This application **does not execute trades**.  It is intended as an educational tool to assist manual decision making.  You retain full control over whether and how to act on the provided signals.

## Prerequisites

1. **Interactive Brokers account** – Create an IBKR account if you do not already have one.  This is free and gives access to the Trader Workstation (TWS) platform and its API.  Sign up at <https://www.interactivebrokers.com>.

2. **Trader Workstation (TWS) or IB Gateway installed** – Download and install the IBKR Trader Workstation or the headless IB Gateway from the Interactive Brokers website.  Either application can provide the API connection used by this project.

3. **API access enabled** – Launch TWS (or Gateway), log in, and enable API access:
   - Open **`File → Global Configuration → API → Settings`**.
   - Tick **“Enable ActiveX and Socket Clients”**.
   - Add `127.0.0.1` to the **Trusted IPs** list.
   - Note the **port number** (default is `7497` for paper trading, `7496` for a live account) and leave clientId at `1`.

4. **Python 3.10 or newer** – Install Python on your machine from <https://www.python.org/downloads/>.

5. **Node.js (optional)** – Only required if you wish to develop a more complex frontend.  The provided example uses plain HTML/JS and does not require Node.js.

## Installation

Open a terminal and navigate to the `trading_app` directory.  Install the Python dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Configuration

You can configure the IBKR connection by exporting the following environment variables before running the server:

- `IBKR_HOST` – Host name of the machine running TWS or IB Gateway (usually `127.0.0.1`).
- `IBKR_PORT` – Socket port configured in TWS/Gateway (`7497` for paper trading by default).
- `IBKR_CLIENT_ID` – Unique client ID for your API session (`1` by default).

If these variables are not set the application will fall back to `127.0.0.1:7497` with client ID `1`.

## Running the backend

Start TWS or IB Gateway and ensure it is logged in and the API is enabled.  Then start the FastAPI server:

```bash
uvicorn main:app --reload --port 8000 --host 0.0.0.0
```

This launches the backend on `http://localhost:8000`.  On startup it will connect to IBKR, fetch bars for each symbol in the default watchlist (`TQQQ`, `SPY`, `QQQ`) and compute a suite of indicators (RSI, MACD, moving averages, Bollinger Bands, ATR, Stochastic oscillator) as well as classify the current candlestick.  It will then update every minute in the background.

### API endpoints

* `GET /signals` – Returns a list of all watched symbols and their latest indicator values and signals.
* `GET /signal/{symbol}` – Returns data for a single symbol in the watchlist.
* `POST /watch/{symbol}` – Adds a new symbol to the watchlist.
* `DELETE /watch/{symbol}` – Removes a symbol from the watchlist.
* `GET /scan?limit={n}` – Performs a basic “most active stocks” scan and returns up to `n` symbols.  You can add any of these suggestions to your watchlist by calling the POST endpoint.

These endpoints all return JSON data and are used by the frontend.

## Running the frontend

The provided frontend is a static HTML/JavaScript file located in `trading_app/frontend/index.html`.  You can open this file directly in your browser (e.g. double click it) or serve it via a simple HTTP server.  It expects the backend to be running on the same host and port (`http://localhost:8000`).  If you run the backend on another host or port you should edit the `API_BASE` constant near the top of the `<script>` in `index.html`.

### Features

- **Comprehensive metrics** – Real‑time table of watched symbols with columns for price, RSI, MACD and its signal/histogram, ATR, Stochastic oscillator values and the detected candlestick pattern.  These are all computed locally from the OHLC data provided by IBKR.
- **Candlestick‑aware signals** – Rows are colour coded green for BUY and red for SELL.  A BUY signal requires oversold RSI (<30), bullish MACD and a bullish candle pattern (strong candle or hammer).  A SELL signal requires overbought RSI (>70), bearish MACD and a bearish candle pattern (strong candle or shooting star).  A progress bar still visualises how close the RSI is to the extreme thresholds.
- **Watchlist management** – Input field to add additional ticker symbols to the watchlist on the fly and a button to remove symbols.
- **Scanner integration** – A simple scanner uses the IBKR API to fetch the most active US stocks; click a suggestion to add it to the watchlist.

## Extending the application

This project is intentionally minimal to make it easy to understand and customise.  Possible extensions include:

* **Additional indicators** – Add moving averages, Bollinger Bands, volume analysis or any other technical measures by extending `compute_indicators()` in `main.py`.
* **Signal refinement** – Improve the trading logic beyond simple RSI/MACD crossovers; include price action, support/resistance levels or machine learning models.
* **Persistent storage** – Save watchlists and signal history to a database (e.g. SQLite) rather than keeping them in memory.
* **User authentication** – Add login protection so that your dashboard is not open to everyone.
* **Deployment** – Package the backend and frontend into a Docker container and deploy to a server or cloud platform.

## Troubleshooting

* If the server fails to connect to IBKR, ensure TWS or Gateway is running, logged in and the API socket is enabled.  Check that the `IBKR_PORT` value matches the port configured in TWS.
* If no data appears for a symbol, verify that the instrument is available through IBKR and that your market data subscription includes it.  Some instruments require additional data subscriptions.
* When adding many symbols at once the free data tier may throttle requests.  Increase the update interval or limit the number of instruments you watch simultaneously.

## Disclaimer

This software is provided for educational purposes only and does not constitute financial advice.  Trading involves risk, and past performance of any strategy does not guarantee future results.  You are solely responsible for any trades you place using information derived from this software.
