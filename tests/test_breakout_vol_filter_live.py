import unittest
from unittest.mock import patch
import pandas as pd
from app.strategies import breakout_vol_filter_live as strategy
from app.broker.alpaca_trade_intent import TradeIntent

class BreakoutSignalTests(unittest.TestCase):
    def _frames(self):
        idx = pd.to_datetime(["2026-10-26", "2026-10-27"])
        raw = pd.DataFrame({"Open":[99,100],"High":[101,102],"Low":[98,99],"Close":[100,101],"Volume":[1000,1500]},index=idx)
        return raw.copy(), raw.copy()

    def _prepared(self, raw, is_spy=False, volume_pass=True, vol_pass=True):
        out=raw.copy()
        out["sma20"]=[95,99]
        out["sma50"]=[90,95]
        out["prior_high20"]=[99,100] if not is_spy else [450,490]
        out["vol_avg20"]=[1000,1000]
        out["atr14"]=[2,2]
        out["rsi14"]=[55,55]
        out["realized_vol20"]=[0.10,0.10 if vol_pass else 0.30]
        out["volatility_limit80"]=[0.20,0.20]
        if is_spy:
            out["Close"]=[495,500]
            out["sma50"]=[480,490]
        else:
            out["Close"]=[100,101]
            out["sma20"]=[95,99]
            out["sma50"]=[90,95]
            out["prior_high20"]=[99,100]
            out["Volume"]=[1000,1500 if volume_pass else 1000]
        return out

    def test_missing_history_fails_closed(self):
        result=strategy.signal_from_completed_bars(pd.DataFrame(),pd.DataFrame())
        self.assertEqual(result["signal"],"WATCH")

    def test_all_breakout_filters_pass(self):
        ticker, spy = self._frames()
        with patch.object(strategy, "indicators", side_effect=[self._prepared(ticker), self._prepared(spy,is_spy=True)]):
            result=strategy.signal_from_completed_bars(ticker,spy)
        self.assertEqual(result["signal"],"BUY")
        self.assertTrue(all(result["checks"].values()))

    def test_spy_volatility_filter_blocks(self):
        ticker, spy = self._frames()
        with patch.object(strategy, "indicators", side_effect=[self._prepared(ticker), self._prepared(spy,is_spy=True,vol_pass=False)]):
            result=strategy.signal_from_completed_bars(ticker,spy)
        self.assertEqual(result["signal"],"WATCH")
        self.assertFalse(result["checks"]["spy_volatility_filter"])

    def test_intent_rejects_partial_bracket(self):
        with self.assertRaises(ValueError):
            TradeIntent(symbol="NVDA",side="buy",quantity=1,confidence=75,signal="BUY",stop_loss_price=100)

    def test_intent_accepts_valid_bracket(self):
        intent=TradeIntent(symbol="NVDA",side="buy",quantity=1,confidence=75,signal="BUY",stop_loss_price=95,take_profit_price=110)
        self.assertEqual(intent.stop_loss_price,95)
        self.assertEqual(intent.take_profit_price,110)

if __name__ == "__main__": unittest.main()
