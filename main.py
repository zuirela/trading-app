from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ib_insync import IB, ScannerSubscription, Stock

app = FastAPI(
    title="Trading Signal Service",
    version="1.2.0",
    description="RSI / MACD / Bollinger / ATR / Stochastic + entry scoring (watchlist only)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

watchlist: List[str] = []
signals: Dict[str, Dict[str, Any]] = {}
ib: Optional[IB] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))

    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
    macd_line = ema(df["close"], 12) - ema(df["close"], 26)
    macd_signal = ema(macd_line, 9)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_line - macd_signal

    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["sma_20"] = sma20
    df["bollinger_upper"] = sma20 + 2 * std20
    df["bollinger_lower"] = sma20 - 2 * std20

    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    k = (df["close"] - low14) / (high14 - low14) * 100
    df["stoch_k"] = k
    df["stoch_d"] = k.rolling(3).mean()

    return df


def score_entry(last: pd.Series) -> Dict[str, Any]:
    close = float(last["close"])
    rsi = float(last.get("rsi", np.nan))
    macd = float(last.get("macd", 0))
    macds = float(last.get("macd_signal", 0))
    hist = float(last.get("macd_hist", 0))
    k = float(last.get("stoch_k", 0))
    d = float(last.get("stoch_d", 0))
    upper = float(last.get("bollinger_upper", close))
    lower = float(last.get("bollinger_lower", close))
    mid = float(last.get("sma_20", close))
    atr = float(last.get("atr", 0)) or (close * 0.01)

    score = 0
    side = "WAIT"
    reasons: List[str] = []

    if not np.isnan(rsi):
        if rsi <= 30:
            score += 25
            side = "LONG"
            reasons.append("RSI oversold")
        elif rsi >= 70:
            score += 25
            side = "SHORT"
            reasons.append("RSI overbought")

    if hist > 0 and macd > macds:
        score += 20
    if hist < 0 and macd < macds:
        score += 20

    if k < 20:
        score += 15
    if k > 80:
        score += 15

    score = int(min(100, score))

    entry_px = close
    stop = close - 2 * atr
    tp = close + 2 * atr
    if side == "SHORT":
        stop, tp = close + 2 * atr, close - 2 * atr

    return {
        "entry_side": side,
        "entry_score": score,
        "entry_reason": ", ".join(reasons) or "Neutral",
        "entry_price": float(entry_px),
        "stop": float(stop),
        "take_profit": float(tp),
        "rsi": float(rsi) if not np.isnan(rsi) else None,
        "macd": float(macd),
        "macd_signal": float(macds),
        "macd_hist": float(hist),
        "stoch_k": float(k),
        "stoch_d": float(d),
        "sma_20": float(mid),
        "bollinger_upper": float(upper),
        "bollinger_lower": float(lower),
        "atr": float(atr),
    }


async def _fetch_bars(contract, duration: str = "1 D", bar_size: str = "5 mins"):
    for what in ("TRADES", "MIDPOINT"):
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what,
            useRTH=True,
            formatDate=1,
        )
        if bars:
            return bars
    return []


async def compute_signal_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    if ib is None or not ib.isConnected():
        return None
    try:
        bars = await _fetch_bars(Stock(symbol, "SMART", "USD"))
        if not bars:
            return None
        df = pd.DataFrame(bars)[["date", "open", "high", "low", "close", "volume"]].copy()
        df = compute_indicators(df)
        if df.empty:
            return None
        last = df.iloc[-1]
        entry = score_entry(last)
        return {
            "symbol": symbol,
            "price": float(last["close"]),
            "time": str(last["date"]),
            **entry,
        }
    except Exception as e:
        print("compute_signal_for_symbol error:", symbol, e)
        return None


async def update_symbol(symbol: str):
    if symbol not in watchlist:
        return
    sig = await compute_signal_for_symbol(symbol)
    if sig:
        signals[symbol] = sig


async def ensure_connected() -> bool:
    global ib
    if ib and ib.isConnected():
        return True
    if ib is None:
        ib = IB()
    try:
        await ib.connectAsync(
            os.environ.get("IBKR_HOST", "127.0.0.1"),
            int(os.environ.get("IBKR_PORT", "7496")),
            clientId=int(os.environ.get("IBKR_CLIENT_ID", "7")),
        )
        print("Connected to IBKR API")
        return True
    except Exception as e:
        print("IB connect failed:", e)
        return False


@app.on_event("startup")
async def startup():
    ok = await ensure_connected()
    if ok:
        asyncio.create_task(update_loop())


@app.on_event("shutdown")
async def shutdown():
    global ib
    if ib and ib.isConnected():
        ib.disconnect()


async def update_loop():
    while True:
        try:
            await asyncio.gather(*(update_symbol(s) for s in list(watchlist)))
        except Exception as e:
            print("Background loop error:", e)
        await asyncio.sleep(60)


@app.get("/signals")
async def get_signals() -> List[Dict[str, Any]]:
    data = [signals[s] for s in watchlist if s in signals]
    data.sort(key=lambda x: x.get("entry_score", 0), reverse=True)
    return data


@app.get("/signal/{symbol}")
async def get_signal(symbol: str) -> Dict[str, Any]:
    sym = symbol.upper()
    if sym not in watchlist:
        raise HTTPException(404, "Symbol not in watchlist")
    if sym not in signals:
        await update_symbol(sym)
    return signals.get(sym, {})


@app.post("/watch/{symbol}")
async def add_symbol(symbol: str):
    sym = symbol.upper()
    if sym not in watchlist:
        watchlist.append(sym)
        asyncio.create_task(update_symbol(sym))
    return {"message": f"Added {sym}"}


@app.delete("/watch/{symbol}")
async def delete_symbol(symbol: str):
    sym = symbol.upper()
    if sym in watchlist:
        watchlist.remove(sym)
        signals.pop(sym, None)
        return {"message": f"Removed {sym}"}
    raise HTTPException(404, "Symbol not in watchlist")


@app.get("/scan")
async def scan(limit: int = 3, q: Optional[str] = None):
    ok = await ensure_connected()
    if not ok:
        return []

    sub = ScannerSubscription(
        instrument="STK", locationCode="STK.US.MAJOR", scanCode="MOST_ACTIVE"
    )
    try:
        res = await ib.reqScannerDataAsync(sub)
    except Exception as e:
        print("Scanner error:", e)
        return []

    items: List[Dict[str, Any]] = []
    for r in res:
        try:
            sym = r.contractDetails.contract.symbol
            name = r.contractDetails.longName or ""
            if q and (q.upper() not in sym.upper() and q.upper() not in name.upper()):
                continue
            sig = await compute_signal_for_symbol(sym)
            if not sig:
                continue
            items.append({
                "symbol": sym,
                "description": name,
                "entry_score": sig.get("entry_score", 0),
                "entry_side": sig.get("entry_side", "WAIT"),
                **{k: sig.get(k) for k in [
                    "rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
                    "sma_20", "bollinger_upper", "bollinger_lower", "atr"
                ]}
            })
        except Exception as e:
            print("scan enrich error:", e)

    items.sort(key=lambda x: x.get("entry_score", 0), reverse=True)
    return items[: max(1, limit)]
