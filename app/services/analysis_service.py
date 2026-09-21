"""AI Trader - Analysis Service V1.0."""
from __future__ import annotations
from typing import Any, Dict, Optional
from app.scanners.watchlist_scanner_v2_1_1 import scan_buy_opportunity
VERSION = "1.0"
SERVICE = "Analysis Service"

def analyze_ticker(ticker: str, period: str = "5y", interval: str = "1d", benchmark: Optional[str] = None) -> Dict[str, Any]:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return {"service": SERVICE, "version": VERSION, "ticker": "", "status": "ERROR", "final_signal": "ERROR", "error": "Ticker is required."}
    result = scan_buy_opportunity({"ticker": symbol, "period": period, "interval": interval, "benchmark": benchmark, "buy_target": 0.0})
    result = dict(result)
    result["service"] = SERVICE
    result["service_version"] = VERSION
    return result
