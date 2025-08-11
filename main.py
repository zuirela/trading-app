"""
FastAPI backend for a simple trading signal web application.

This backend connects to Interactive Brokers (IBKR) Trader Workstation or
IB Gateway via the ib_insync library to fetch real‑time market data and
computes technical indicators (RSI and MACD) plus extra metrics from
OHLCV candles. It exposes REST API endpoints to retrieve the latest
signals, add/remove symbols from the watchlist and perform a basic scan
for new trading candidates.

The application maintains an in‑memory dictionary of the most recent
statistics for each symbol. A background task refreshes this dictionary
every 60 seconds. When a symbol's RSI drops below 30 while the MACD line
is above its signal line together with a bullish candle pattern, the API
returns a "BUY" recommendation; if the RSI rises above 70 and the MACD
line falls below its signal line together with a bearish candle pattern,
a "SELL" recommendation is returned.
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ib_insync import IB, Stock, ScannerSubscription
import pandas as pd
import numpy as np

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="Trading Signal Service",
    description="Compute RSI/MACD + candle signals from IBKR data",
    version="0.2.0",
)

# CORS: salli selainkäyttö paikallisesti
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Seurantalista (voi muuttaa API:lla)
watchlist: List[str] = ["TQQQ", "SPY", "QQQ"]

# Viimeisimmät signaalit per symboli
signals: Dict[str, Dict[str, Optional[float]]] = {}

# IB‑yhteysobjekti
ib: Optional[IB] = None


# -------------------------------------------------
# Indikaattorit (pandas + numpy, ei ulkoisia TA‑kirjastoja)
# -------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # RSI (14)
    window = 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi

    # EMA helper
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    # MACD (12,26,9)
    ema_fast = ema(df["close"], 12)
    ema_slow = ema(df["close"], 26)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, 9)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_line - macd_signal

    # SMA20 + Bollinger
    sma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["sma_20"] = sma20
    df["bollinger_upper"] = sma20 + 2 * std20
    df["bollinger_lower"] = sma20 - 2 * std20

    # ATR(14)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()

    # Stochastic
    low14 = df["low"].rolling(window=14).min()
    high14 = df["high"].rolling(window=14).max()
    stoch_k = (df["close"] - low14) / (high14 - low14) * 100
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_k.rolling(window=3).mean()

    # Candle‑metriikat ja luokittelu
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    rng_nonzero = rng.replace(0, np.nan)
    body_ratio = body / rng_nonzero
    lower_shadow = df[["close", "open"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["close", "open"]].max(axis=1)
    direction = np.where(
        df["close"] > df["open"],
        "bullish",
        np.where(df["close"] < df["open"], "bearish", "neutral"),
    )

    patterns: List[str] = []
    for i in range(len(df)):
        pat = "neutral"
        if pd.isna(body_ratio.iloc[i]) or pd.isna(rng.iloc[i]):
            patterns.append(pat)
            continue
        br = body_ratio.iloc[i]
        dirn = direction[i]
        ls = lower_shadow.iloc[i]
        us = upper_shadow.iloc[i]
        b = body.iloc[i]

        if br < 0.1:
            pat = "doji"
        elif ls > 2 * b and us <= b:
            pat = "hammer" if dirn == "bullish" else "inverted_hammer"
        elif us > 2 * b and ls <= b:
            pat = "shooting_star" if dirn == "bearish" else "inverted_shooting_star"
        elif br > 0.6:
            pat = "bullish_strong" if dirn == "bullish" else "bearish_strong"
        patterns.append(pat)

    df["candle_pattern"] = patterns
    df["candle_body"] = body
    df["candle_range"] = rng
    df["candle_lower_shadow"] = lower_shadow
    df["candle_upper_shadow"] = upper_shadow
    df["candle_direction"] = direction

    return df


# -------------------------------------------------
# Datapäivitykset
# -------------------------------------------------
async def update_symbol(symbol: str, duration: str = "1 D", bar_size: str = "1 min") -> None:
    """Hae historialliset palkit ja päivitä signaalit."""
    global signals, ib
    if ib is None or not ib.isConnected():
        return

    contract = Stock(symbol, "SMART", "USD")
    try:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )
        df = pd.DataFrame(bars)
        if df.empty:
            return

        # ib_insync antaa attribuutit 'date','open','high','low','close','volume'
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = compute_indicators(df)
        last = df.iloc[-1]

        rsi_val = float(last["rsi"])
        macd_val = float(last["macd"])
        macd_sig = float(last["macd_signal"])
        pattern = str(last.get("candle_pattern", "neutral"))

        recommendation: Optional[str] = None
        if rsi_val < 30 and macd_val > macd_sig and pattern in ("bullish_strong", "hammer"):
            recommendation = "BUY"
        elif rsi_val > 70 and macd_val < macd_sig and pattern in ("bearish_strong", "shooting_star"):
            recommendation = "SELL"

        signals[symbol] = {
            "symbol": symbol,
            "price": float(last["close"]),
            "rsi": rsi_val,
            "macd": macd_val,
            "macd_signal": macd_sig,
            "macd_hist": float(last["macd_hist"]),
            "candle_pattern": pattern,
            "atr": float(last.get("atr", float("nan"))),
            "stoch_k": float(last.get("stoch_k", float("nan"))),
            "stoch_d": float(last.get("stoch_d", float("nan"))),
            "signal": recommendation,
            "time": str(last["date"]),
        }
    except Exception as exc:
        print(f"Error updating {symbol}: {exc}")


async def update_loop() -> None:
    """Päivitä kaikki symbolit jatkuvasti taustalla."""
    UPDATE_INTERVAL = 60  # sekuntia
    while True:
        try:
            await asyncio.gather(*(update_symbol(sym) for sym in watchlist))
        except Exception as exc:
            print("Background update error:", exc)
        await asyncio.sleep(UPDATE_INTERVAL)


# -------------------------------------------------
# Sovelluksen elinkaarikoukut
# -------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    global ib
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PORT", "7497"))
    client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))

    ib = IB()
    try:
        await ib.connectAsync(host, port, clientId=client_id)
        print("Connected to IBKR API")
    except Exception as e:
        print("API connection failed:", repr(e))
        raise

    # nollaa signaalit ja tee heti eka päivitys
    for sym in watchlist:
        await update_symbol(sym)

    # käynnistä taustasilmukka
    asyncio.create_task(update_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    global ib
    if ib is not None and ib.isConnected():
        ib.disconnect()


# -------------------------------------------------
# REST‑endpointit
# -------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/signals")
async def get_signals() -> List[Dict[str, Optional[float]]]:
    return list(signals.values())

@app.get("/signal/{symbol}")
async def get_signal(symbol: str) -> Dict[str, Optional[float]]:
    sym = symbol.upper()
    if sym not in watchlist:
        raise HTTPException(status_code=404, detail="Symbol not found in watchlist")
    if sym not in signals:
        await update_symbol(sym)
    return signals.get(sym, {})

@app.post("/watch/{symbol}")
async def add_symbol(symbol: str) -> Dict[str, str]:
    sym = symbol.upper()
    if sym in watchlist:
        return {"message": f"{sym} is already being watched"}
    watchlist.append(sym)
    await update_symbol(sym)
    return {"message": f"Added {sym} to watchlist"}

@app.delete("/watch/{symbol}")
async def remove_symbol(symbol: str) -> Dict[str, str]:
    sym = symbol.upper()
    if sym not in watchlist:
        raise HTTPException(status_code=404, detail="Symbol not in watchlist")
    watchlist.remove(sym)
    signals.pop(sym, None)
    return {"message": f"Removed {sym} from watchlist"}

@app.get("/scan")
async def scan_most_active(limit: int = 5) -> List[Dict[str, str]]:
    """IBKR-scanner: MOST_ACTIVE Yhdysvaltain osakkeet."""
    global ib
    if ib is None or not ib.isConnected():
        return []
    sub = ScannerSubscription(
        instrument="STK",
        locationCode="STK.US.MAJOR",
        scanCode="MOST_ACTIVE",
    )
    try:
        res = await ib.reqScannerDataAsync(sub)
    except Exception as e:
        print("Scanner error:", e)
        return []
    out = []
    for r in res[:limit]:
        out.append({"symbol": r.contract.symbol, "description": r.description})
    return out
