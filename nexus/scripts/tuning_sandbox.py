import os
import sys
import pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from nexus.core.signal_engine import TechnicalSignalEngine

def main():
    print("=" * 70)
    print(" 🏦 NEXUS ALPHA v3 — INSTITUTIONAL MEAN-REVERSION SANDBOX")
    print("=" * 70)
    
    MODE = "binary"
    MIN_CONSENSUS = 1     # Irrelevante para Binary (pipeline directo)
    BARS_TO_TEST = 7000   # Usar TODA la data disponible
    # El Alpha v3 genera confianza directa (score compuesto).
    # Threshold de 0.55 = BB Touch + RSI Extreme activados.
    TARGET_CONFIDENCE = 0.60
    
    if MODE == "binary":
        data_path = os.path.join("nexus", "data", "vault", "BTCUSD", "5m.csv")
    else:
        data_path = os.path.join("nexus", "data", "vault", "BTC-USD", "15m.csv")
        
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró data local en {data_path}")
        return
        
    print(f"📂 Data: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"📊 Total barras disponibles: {len(df)}")
    
    engine = TechnicalSignalEngine(mode=MODE, min_consensus=MIN_CONSENSUS)

    limit = min(BARS_TO_TEST, len(df))
    print(f"🎯 Umbral de confianza: {TARGET_CONFIDENCE}")
    print(f"⏳ Analizando {limit} velas...\n")
    
    total_signals = 0
    valid_trades = 0
    signals_conf = []
    expiry_stats = {}
    
    for i in range(100, limit):
        sub_df = df.iloc[i-100:i+1]
        res = engine.generate_signal(sub_df)
        
        signal = res.get("signal", "").upper() if isinstance(res.get("signal"), str) else ""
        conf = res.get("confidence", 0.0)
        
        if signal in ("BUY", "SELL"):
            total_signals += 1
            signals_conf.append(conf)
            
            if conf >= TARGET_CONFIDENCE:
                valid_trades += 1
                tz = sub_df.index[-1].strftime("%Y-%m-%d %H:%M") if hasattr(sub_df.index[-1], 'strftime') else str(sub_df.index[-1])
                
                # Simulación multi-expiry
                for exp_bars in [2, 3, 4, 5]:
                    future_idx = min(i + exp_bars, len(df) - 1)
                    entry_price = df["close"].iloc[i]
                    exit_price = df["close"].iloc[future_idx]
                    
                    if signal == "BUY":
                        win = exit_price > entry_price
                    else:
                        win = exit_price < entry_price
                    
                    exp_key = f"exp_{exp_bars}"
                    if exp_key not in expiry_stats:
                        expiry_stats[exp_key] = {"wins": 0, "losses": 0}
                    if win:
                        expiry_stats[exp_key]["wins"] += 1
                    else:
                        expiry_stats[exp_key]["losses"] += 1
                    
                    # También separar stats para conf >= 0.70
                    if conf >= 0.70:
                        hc_key = f"hc_{exp_bars}"
                        if hc_key not in expiry_stats:
                            expiry_stats[hc_key] = {"wins": 0, "losses": 0}
                        if win:
                            expiry_stats[hc_key]["wins"] += 1
                        else:
                            expiry_stats[hc_key]["losses"] += 1

    print(f"\n{'=' * 70}")
    print(f" 📊 RESULTADOS DEL SANDBOX ({limit} Barras)")
    print(f"{'=' * 70}")
    print(f"🔸 Señales crudas totales (cualquier confianza): {total_signals}")
    print(f"🚀 Trades válidos (Confianza >= {TARGET_CONFIDENCE}): {valid_trades}")
    
    if valid_trades > 0 and expiry_stats:
        payout = 0.85
        breakeven = 1 / (1 + payout) * 100
        
        print(f"\n   ═══ TODOS LOS TRADES (Confianza >= {TARGET_CONFIDENCE}) ═══")
        print(f"   {'Expiry':<10} {'Wins':<8} {'Losses':<8} {'WinRate':<10} {'EV/Trade':<12} {'Status'}")
        print(f"   {'─' * 60}")
        
        for exp_key in sorted([k for k in expiry_stats.keys() if k.startswith("exp_")]):
            stats = expiry_stats[exp_key]
            w, l = stats["wins"], stats["losses"]
            total = w + l
            wr = (w / total) * 100 if total > 0 else 0
            ev = (wr/100 * payout) - ((100 - wr)/100 * 1.0)
            viable = "🏆 RENTABLE" if ev > 0 else "⚠️"
            bars = exp_key.split("_")[1]
            print(f"   {bars+'m':<10} {w:<8} {l:<8} {wr:.1f}%{'':>4} {ev*100:+.2f}%{'':>5} {viable}")
        
        hc_keys = sorted([k for k in expiry_stats.keys() if k.startswith("hc_")])
        if hc_keys:
            print(f"\n   ═══ SOLO ALTA CONFIANZA (Confianza >= 0.70) ═══")
            print(f"   {'Expiry':<10} {'Wins':<8} {'Losses':<8} {'WinRate':<10} {'EV/Trade':<12} {'Status'}")
            print(f"   {'─' * 60}")
            for hc_key in hc_keys:
                stats = expiry_stats[hc_key]
                w, l = stats["wins"], stats["losses"]
                total = w + l
                wr = (w / total) * 100 if total > 0 else 0
                ev = (wr/100 * payout) - ((100 - wr)/100 * 1.0)
                viable = "🏆 RENTABLE" if ev > 0 else "⚠️"
                bars = hc_key.split("_")[1]
                print(f"   {bars+'m':<10} {w:<8} {l:<8} {wr:.1f}%{'':>4} {ev*100:+.2f}%{'':>5} {viable}")
        
        print(f"\n   📌 Breakeven con Payout {payout*100:.0f}%: {breakeven:.1f}% Win Rate")
    
    print(f"\nDistribución de Confianza:")
    if signals_conf:
        s = pd.Series(signals_conf)
        print(s.value_counts().sort_index(ascending=False).to_string())
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
