"""
NEXUS Alpha v3 — Multi-Asset Multi-Timeframe Scanner
Prueba la estrategia con TODOS los activos y timeframes disponibles.
"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nexus.core.signal_engine import TechnicalSignalEngine

def test_asset(data_path, label, mode="binary"):
    if not os.path.exists(data_path):
        print(f"  ❌ {label}: Archivo no encontrado")
        return
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    bars = len(df)
    
    # Check data quality
    hl_range = df['high'] - df['low']
    zero_pct = (hl_range == 0).sum() / bars * 100
    
    engine = TechnicalSignalEngine(mode=mode, min_consensus=1)
    
    limit = min(5000, bars)
    signals = []
    
    for i in range(100, limit):
        sub = df.iloc[i-100:i+1]
        res = engine.generate_signal(sub)
        sig = res.get("signal", "").upper()
        conf = res.get("confidence", 0.0)
        
        if sig in ("BUY", "SELL") and conf >= 0.60:
            # Test expiry de 5 barras
            future_idx = min(i + 5, bars - 1)
            entry = float(df["close"].iloc[i])
            exit_p = float(df["close"].iloc[future_idx])
            win = (exit_p > entry) if sig == "BUY" else (exit_p < entry)
            signals.append({"sig": sig, "conf": conf, "win": win})
    
    total = len(signals)
    if total == 0:
        print(f"  ⚪ {label:<25} | {bars:>5} bars | Zero-range: {zero_pct:.0f}% | 0 señales")
        return
    
    wins = sum(1 for s in signals if s["win"])
    wr = wins / total * 100
    payout = 0.85
    ev = (wr/100 * payout) - ((100 - wr)/100)
    status = "🏆 RENTABLE" if ev > 0 else "⚠️"
    
    # Also test high-conf only
    hc = [s for s in signals if s["conf"] >= 0.70]
    hc_total = len(hc)
    hc_wr = (sum(1 for s in hc if s["win"]) / hc_total * 100) if hc_total > 0 else 0
    
    print(f"  {status} {label:<25} | {bars:>5} bars | ZR: {zero_pct:.0f}% | Trades: {total:>4} | WR: {wr:.1f}% | HC({hc_total}): {hc_wr:.1f}% | EV: {ev*100:+.1f}%")

def main():
    print("=" * 90)
    print(" 🏦 NEXUS ALPHA — MULTI-ASSET SCANNER (5-bar expiry, Payout 85%)")
    print("=" * 90)
    
    vault = os.path.join("nexus", "data", "vault")
    
    tests = [
        (os.path.join(vault, "EURUSDX", "1m.csv"),  "EURUSD 1m",   "binary"),
        (os.path.join(vault, "BTCUSD", "5m.csv"),    "BTCUSD 5m",   "binary"),
        (os.path.join(vault, "BTCUSD", "15m.csv"),   "BTCUSD 15m",  "binary"),
        (os.path.join(vault, "BTCUSD", "1h.csv"),    "BTCUSD 1h",   "binary"),
        (os.path.join(vault, "BTCUSDT", "5m.csv"),   "BTCUSDT 5m",  "binary") if os.path.exists(os.path.join(vault, "BTCUSDT", "5m.csv")) else None,
        (os.path.join(vault, "BTCUSDT", "15m.csv"),  "BTCUSDT 15m", "binary") if os.path.exists(os.path.join(vault, "BTCUSDT", "15m.csv")) else None,
    ]
    
    for t in tests:
        if t:
            test_asset(*t)
    
    print("=" * 90)
    print(" 📌 Breakeven = 54.1% WR | HC = High Confidence (0.70+) | ZR = Zero-Range candles")
    print("=" * 90)

if __name__ == "__main__":
    main()
