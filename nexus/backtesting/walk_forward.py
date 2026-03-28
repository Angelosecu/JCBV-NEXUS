"""
NEXUS Trading System — Walk-Forward Optimization Engine v2
===========================================================
Fase 12 → v2: WFO Heurístico Puro para Calibración Eficiente.

CAMBIOS v2:
  - NexusWFOStrategy: Estrategia de calibración pura SIN LLM.
    Usa señales técnicas directamente con umbrales de confianza
    para generar trades reales en cada ventana IS/OOS.
  - _extract_best_params: Selección por MEDIA de OOS Sharpe positivos
    en todas las ventanas (no solo la última).
  - Formato de consola unificado: [SYMBOL | TF | MODE] en cada línea.
  - Tabla de resumen profesional al finalizar cada ejecución WFO.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import time
from datetime import timedelta
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np

try:
    import backtrader as bt
except ImportError:
    bt = None  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("nexus.wfo")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ══════════════════════════════════════════════════════════════════════
#  WFO-Mode Strategy → Puro Heurístico, Sin LLM, Sin Cache
#  Propósito: calibración de hiperparámetros de riesgo/reward eficiente
# ══════════════════════════════════════════════════════════════════════

class NexusWFOStrategy(bt.Strategy):  # type: ignore
    """
    Estrategia de calibración WFO sin LLM.

    Genera señales directamente desde el TechnicalSignalEngine
    usando un umbral de consenso. Esto garantiza que haya trades
    reales para medir el impacto de SL/TP/Confidence en cada ventana.

    NO usa: LLM, global cache, async calls.
    USA:    Señales técnicas puras + ATR stops institucionales.
    """

    params = dict(  # type: ignore
        atr_period=14,
        sl_atr_mult=2.0,
        tp_atr_mult=3.0,
        min_confidence=0.65,
        lookback=100,
        signal_threshold=2,   # 2 de 4 indicadores activos (EMA Cross inactivo con lookback=60)
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)  # type: ignore

        # Lazy import de señales técnicas
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from core.signal_engine import TechnicalSignalEngine  # type: ignore
        self.signal_engine = TechnicalSignalEngine(mode="spot", min_consensus=self.p.signal_threshold)

        # Suprimir logging verbose del signal_engine durante WFO
        # (evita millones de líneas "Señal generada: HOLD" por cada barra)
        logging.getLogger("nexus.signal_engine").setLevel(logging.WARNING)

        self._trade_log: List[Dict[str, Any]] = []
        self._equity_curve: List[float] = []

    def next(self):
        self._equity_curve.append(self.broker.getvalue())

        # Esperar suficientes barras para indicadores
        if len(self.data) < self.p.lookback:
            return

        # Construir DataFrame rápido para señales
        n = min(len(self.data), self.p.lookback)
        df = pd.DataFrame({
            "open":   [self.data.open[-i] for i in range(n - 1, -1, -1)],
            "high":   [self.data.high[-i] for i in range(n - 1, -1, -1)],
            "low":    [self.data.low[-i] for i in range(n - 1, -1, -1)],
            "close":  [self.data.close[-i] for i in range(n - 1, -1, -1)],
            "volume": [self.data.volume[-i] for i in range(n - 1, -1, -1)],
        })

        try:
            technical = self.signal_engine.generate_signal(df)
        except Exception:
            return

        confidence = technical.get("confidence", 0.0)
        signal = technical.get("signal", "")
        action = None
        
        signal_str = signal.upper() if isinstance(signal, str) else getattr(signal, "name", "").upper()
        
        if signal_str.startswith("BUY"):
            action = "BUY"
        elif signal_str.startswith("SELL"):
            action = "SELL"
            
        # Gate de confianza mínima
        if confidence < self.p.min_confidence or action is None:
            return

        # Evitar abrir si ya en posición
        if self.position:
            return

        current_price = self.data.close[0]
        atr_val = self.atr[0]

        if action == "BUY":
            sl_price = current_price - (self.p.sl_atr_mult * atr_val)
            tp_price = current_price + (self.p.tp_atr_mult * atr_val)
            size = (self.broker.getvalue() * 0.05) / current_price
            if size > 0:
                self.buy_bracket(size=round(size, 6), stopprice=sl_price, limitprice=tp_price)
                self._trade_log.append({
                    "action": "BUY", "price": current_price,
                    "sl": sl_price, "tp": tp_price, "confidence": confidence,
                })

        elif action == "SELL":
            sl_price = current_price + (self.p.sl_atr_mult * atr_val)
            tp_price = current_price - (self.p.tp_atr_mult * atr_val)
            size = (self.broker.getvalue() * 0.05) / current_price
            if size > 0:
                self.sell_bracket(size=round(size, 6), stopprice=sl_price, limitprice=tp_price)
                self._trade_log.append({
                    "action": "SELL", "price": current_price,
                    "sl": sl_price, "tp": tp_price, "confidence": confidence,
                })

    def stop(self):
        """Registrar equity final."""
        self._equity_curve.append(self.broker.getvalue())


# ══════════════════════════════════════════════════════════════════════
#  Binary WFO-Mode Strategy → Puro Heurístico, Sin LLM
# ══════════════════════════════════════════════════════════════════════

class NexusBinaryWFOStrategy(bt.Strategy):  # type: ignore
    """
    Estrategia de calibración WFO pura para Opciones Binarias sin LLM.
    Simula la rentabilidad basada en el win-rate de señales técnicas
    conquistando la barra de expiración.
    """

    params = dict(  # type: ignore
        min_confidence=0.70,
        payout_pct=0.85,
        expiry_bars=3,
        stake_pct=1.0,           # Riesgo fijo por trade: 1%
        max_positions=3,
        lookback=100,  # Buffer expandido para indicadores (EMA 50, etc)
        signal_threshold=2,      # Mínimo de indicadores en acuerdo
    )

    def __init__(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from core.signal_engine import TechnicalSignalEngine  # type: ignore
        self.signal_engine = TechnicalSignalEngine(mode="binary", min_consensus=self.p.signal_threshold)
        
        self.binary_equity = 10_000.0  # Ledger contable inicial
        self._active_options: List[Dict[str, Any]] = []
        self._trade_log: List[Dict[str, Any]] = []
        self._equity_curve: List[float] = []
        self.metrics_initialized = False

    def next(self):
        if not self.metrics_initialized:
            self.binary_equity = self.broker.getcash()
            self.metrics_initialized = True

        self._equity_curve.append(self.binary_equity)
        self._evaluate_expirations()

        if len(self.data) < self.p.lookback:
            return

        n = min(len(self.data), self.p.lookback)
        df = pd.DataFrame({
            "open":   [self.data.open[-i] for i in range(n - 1, -1, -1)],
            "high":   [self.data.high[-i] for i in range(n - 1, -1, -1)],
            "low":    [self.data.low[-i] for i in range(n - 1, -1, -1)],
            "close":  [self.data.close[-i] for i in range(n - 1, -1, -1)],
            "volume": [self.data.volume[-i] for i in range(n - 1, -1, -1)],
        })

        try:
            technical = self.signal_engine.generate_signal(df)
        except Exception:
            return

        confidence = technical.get("confidence", 0.0)
        signal = technical.get("signal", "")
        action = None
        
        signal_str = signal.upper() if isinstance(signal, str) else getattr(signal, "name", "").upper()
        
        if signal_str.startswith("BUY"):
            action = "BUY"
        elif signal_str.startswith("SELL"):
            action = "SELL"

        if confidence < self.p.min_confidence or action is None:
            return
            
        if len(self._active_options) >= self.p.max_positions:
            return

        current_price = self.data.close[0]

        if action:
            stake = self.binary_equity * (self.p.stake_pct / 100.0)
            stake = max(stake, 1.0)
            
            if self.binary_equity >= stake:
                self.binary_equity -= stake
                self._active_options.append({
                    "entry_bar": len(self.data),
                    "action": action,
                    "entry_price": current_price,
                    "stake": stake,
                    "payout_pct": self.p.payout_pct,
                    "expiry_bars": self.p.expiry_bars
                })

    def _evaluate_expirations(self):
        current_bar = len(self.data)
        current_price = self.data.close[0]
        surviving_options = []

        for opt in self._active_options:
            if (current_bar - opt["entry_bar"]) >= opt["expiry_bars"]:
                itm = False
                if opt["action"] == "BUY" and current_price > opt["entry_price"]:
                    itm = True
                elif opt["action"] == "SELL" and current_price < opt["entry_price"]:
                    itm = True
                
                if itm:
                    # Gana
                    profit = opt["stake"] * opt["payout_pct"]
                    self.binary_equity += (opt["stake"] + profit)
                else:
                    # Pierde (stake ya fue deducido al abrir)
                    profit = -opt["stake"]
                
                self._trade_log.append({
                    "action": opt["action"],
                    "pnl": profit,
                    "result": "WIN" if itm else "LOSS"
                })
            else:
                surviving_options.append(opt)
                
        self._active_options = surviving_options

    def stop(self):
        # Asegurarse de contabilizar el valor final
        self._equity_curve.append(self.binary_equity)


# ══════════════════════════════════════════════════════════════════════
#  Walk-Forward Optimizer
# ══════════════════════════════════════════════════════════════════════

class WalkForwardOptimizer:
    """Motor de validación institucional rolling-window."""

    def __init__(
        self,
        csv_path: str,
        is_days: float = 90.0,
        oos_days: float = 30.0,
        trading_mode: str = "spot",
    ):
        """
        Args:
            csv_path:     Ruta al archivo CSV procesado.
            is_days:      Días In-Sample (entrenamiento).
            oos_days:     Días Out-of-Sample (validación).
            trading_mode: "spot" o "binary"
        """
        self.csv_path = csv_path
        self.is_days = float(is_days)
        self.oos_days = float(oos_days)
        self.trading_mode = trading_mode

        # Extraer símbolo y timeframe del path para telemetría
        path_parts = os.path.normpath(csv_path).split(os.sep)
        # Estructura vault: ...vault/SYMBOL/TF.csv
        self.symbol = "UNKNOWN"
        self.timeframe = "?"
        try:
            fname = os.path.basename(csv_path).replace(".csv", "").upper()
            # Si viene del vault: csv_path = .../vault/BTCUSD/1h.csv
            parent = os.path.basename(os.path.dirname(csv_path)).upper()
            if parent and parent not in ("VAULT", "HISTORICAL", "DATA"):
                self.symbol = parent
                self.timeframe = fname
            else:
                self.symbol = fname.split("_")[0]
                self.timeframe = fname.split("_")[1] if "_" in fname else "?"
        except Exception:
            pass

        self._prefix = f"[{self.symbol} | {self.timeframe} | {trading_mode.upper()}]"

        logger.info("━" * 64)
        logger.info("%s 📂 Cargando dataset WFO: %s", self._prefix, csv_path)
        self.raw_data = pd.read_csv(
            self.csv_path, parse_dates=["open_time"], index_col="open_time"
        )
        cols = {"open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume"}
        self.raw_data = self.raw_data.rename(columns=cols)
        self.raw_data.sort_index(inplace=True)

        logger.info(
            "%s 📊 Dataset: %d barras | %s → %s",
            self._prefix, len(self.raw_data),
            self.raw_data.index[0].date(), self.raw_data.index[-1].date(),
        )
        logger.info("━" * 64)

    # ──────────────────────────────────────────────────────────────────
    #  Generación de ventanas rolling
    # ──────────────────────────────────────────────────────────────────

    def generate_windows(self) -> List[Dict[str, pd.Timestamp]]:
        """Calcula el rango de cada iteración step-forward."""
        start_date = self.raw_data.index[0]
        end_date = self.raw_data.index[-1]
        windows: List[Dict[str, Any]] = []
        current_start = start_date

        while True:
            is_end = current_start + timedelta(days=self.is_days)
            oos_end = is_end + timedelta(days=self.oos_days)
            if oos_end > end_date:
                break
            windows.append({
                "window_idx": len(windows) + 1,
                "is_start": current_start,
                "is_end": is_end,
                "oos_start": is_end,
                "oos_end": oos_end,
            })
            current_start += timedelta(days=self.oos_days)

        logger.info(
            "%s 🪟 Generadas %d ventanas WFO (IS: %dd | OOS: %dd)",
            self._prefix, len(windows), int(self.is_days), int(self.oos_days),
        )
        return windows

    # ──────────────────────────────────────────────────────────────────
    #  Ejecución de un slice con Backtrader
    # ──────────────────────────────────────────────────────────────────

    def _run_backtrader_slice(
        self, df_slice: pd.DataFrame, params: Dict[str, Any]
    ) -> Tuple[float, List[float], int]:
        """Ejecuta un backtest aislado sobre un segmento de Pandas.
        
        Returns:
            (sharpe, returns_list, trade_count)
        """
        if bt is None:
            raise ImportError("backtrader no está instalado.")

        cerebro = bt.Cerebro(stdstats=False)  # type: ignore
        cerebro.broker.setcash(10_000.0)

        data_feed = bt.feeds.PandasData(  # type: ignore
            dataname=df_slice,
            open="open", high="high", low="low", close="close", volume="volume",
        )
        cerebro.adddata(data_feed)

        if self.trading_mode == "binary":
            cerebro.broker.setcommission(commission=0.0)
            cerebro.addstrategy(
                NexusBinaryWFOStrategy,
                min_confidence=params.get("min_confidence", 0.70),
                payout_pct=0.85,
                expiry_bars=params.get("expiry_bars", 3),
            )
        else:
            # WFO Mode: usa la estrategia heurística pura (sin LLM)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.addstrategy(
                NexusWFOStrategy,
                sl_atr_mult=params.get("sl_atr_mult", 2.0),
                tp_atr_mult=params.get("tp_atr_mult", 3.0),
                min_confidence=params.get("min_confidence", 0.65),
            )

        results = cerebro.run()
        strat = results[0]
        
        # Calcular returns y sharpe desde el equity curve nativo o binario
        returns_list: List[float] = []
        sharpe = 0.0
        
        if hasattr(strat, "_equity_curve") and len(strat._equity_curve) > 1:
            eq = np.array(strat._equity_curve)
            # Ignorar divisiones por cero
            with np.errstate(divide='ignore', invalid='ignore'):
                rets_array = (eq[1:] - eq[:-1]) / eq[:-1]
                # Reemplazar NaNs/infinitos con 0
                rets_array = np.nan_to_num(rets_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            returns_list = rets_array.tolist()
            if len(returns_list) > 1 and np.std(returns_list) > 0:
                sharpe_val = (np.mean(returns_list) / np.std(returns_list, ddof=1)) * np.sqrt(8760)
                sharpe = float(sharpe_val)
        
        trade_count = len(getattr(strat, "_trade_log", []))
        return sharpe, returns_list, trade_count

    # ──────────────────────────────────────────────────────────────────
    #  Pipeline WFO principal
    # ──────────────────────────────────────────────────────────────────

    def execute_wfo(self, param_grid: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ejecuta el pipeline WFO entero.
        Itera ventanas → Busca mejor parámetro IS → Valida OOS → Acumula retornos.
        """
        windows = self.generate_windows()
        oos_equity_curve_stitched = [10_000.0]
        wfo_log: List[Dict[str, Any]] = []
        t_start = time.time()

        for w in windows:
            idx = w["window_idx"]
            total = len(windows)
            is_start = w["is_start"].date()
            is_end = w["is_end"].date()
            oos_start = w["oos_start"].date()
            oos_end = w["oos_end"].date()

            logger.info("")
            logger.info("╔══════════════════════════════════════════════════════════╗")
            logger.info("║  %s  Ventana %d/%d", self._prefix, idx, total)
            logger.info("║  🔬 IS:  %s → %s", is_start, is_end)
            logger.info("║  🧪 OOS: %s → %s", oos_start, oos_end)
            logger.info("╚══════════════════════════════════════════════════════════╝")

            df_is = self.raw_data.loc[w["is_start"]: w["is_end"]]
            df_oos = self.raw_data.loc[w["oos_start"]: w["oos_end"]]

            if len(df_is) < 60 or len(df_oos) < 20:
                logger.warning("%s ⚠️  Ventana %d omitida: datos insuficientes (IS=%d, OOS=%d)",
                               self._prefix, idx, len(df_is), len(df_oos))
                continue

            # ── 1. OPTIMIZACIÓN IN-SAMPLE ──────────────────────────
            best_sharpe = -9999.0
            best_params = param_grid[0]
            best_trades = 0

            for p_idx, params in enumerate(param_grid):
                sharpe, _, trades = self._run_backtrader_slice(df_is, params)
                logger.info(
                    "  %s 🔬 Grid %d/%d: %s → IS Sharpe: %.3f | Trades: %d",
                    self._prefix, p_idx + 1, len(param_grid), params, sharpe, trades,
                )
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_trades = trades

            logger.info(
                "  %s 🏆 [IS] Ganador: %s | Sharpe: %.3f | Trades: %d",
                self._prefix, best_params, best_sharpe, best_trades,
            )

            # ── 2. VALIDACIÓN OUT-OF-SAMPLE ────────────────────────
            oos_sharpe, oos_returns, oos_trades = self._run_backtrader_slice(df_oos, best_params)
            logger.info(
                "  %s 🧪 [OOS] Sharpe: %.3f | Trades: %d | Datos no vistos ✓",
                self._prefix, oos_sharpe, oos_trades,
            )

            # Stitching de equity OOS continua y retorno de ventana
            current_capital = oos_equity_curve_stitched[-1]
            win_ret = 1.0
            for r in oos_returns:
                current_capital *= (1 + r)
                win_ret *= (1 + r)
                oos_equity_curve_stitched.append(current_capital)

            oos_return_pct = (win_ret - 1.0) * 100

            wfo_log.append({
                "window": idx,
                "is_period": f"{is_start} → {is_end}",
                "oos_period": f"{oos_start} → {oos_end}",
                "best_params": best_params,
                "is_sharpe": round(best_sharpe, 4),
                "oos_sharpe": round(oos_sharpe, 4),
                "oos_return_pct": round(oos_return_pct, 2),
                "is_bars": len(df_is),
                "oos_bars": len(df_oos),
                "is_trades": best_trades,
                "oos_trades": oos_trades,
            })

        elapsed = time.time() - t_start

        # ── TABLA RESUMEN PROFESIONAL ──────────────────────────────
        self._print_summary_table(wfo_log, oos_equity_curve_stitched, elapsed)

        return {
            "wfo_log": wfo_log,
            "stitched_equity": oos_equity_curve_stitched,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.trading_mode,
            "elapsed_seconds": round(elapsed, 1),
        }

    # ──────────────────────────────────────────────────────────────────
    #  Tabla de Resumen Profesional
    # ──────────────────────────────────────────────────────────────────

    def _print_summary_table(
        self,
        wfo_log: List[Dict[str, Any]],
        equity: List[float],
        elapsed: float,
    ) -> None:
        """Imprime la tabla institucional de resultados WFO."""
        if not wfo_log:
            logger.warning("%s Sin ventanas completadas.", self._prefix)
            return

        positive_oos = [w["oos_sharpe"] for w in wfo_log if w["oos_sharpe"] > 0] # type: ignore
        negative_oos = [w["oos_sharpe"] for w in wfo_log if w["oos_sharpe"] < 0]
        zero_oos = [w for w in wfo_log if w["oos_sharpe"] == 0.0]
        win_rate = (len(positive_oos) / len(wfo_log)) * 100 if wfo_log else 0
        avg_oos = float(np.mean([w["oos_sharpe"] for w in wfo_log]))
        best_window = max(wfo_log, key=lambda w: w["oos_sharpe"])
        worst_window = min(wfo_log, key=lambda w: w["oos_sharpe"])

        final_equity = equity[-1] if equity else 10_000.0
        total_return = ((final_equity - 10_000.0) / 10_000.0) * 100
        total_trades = sum(w.get("oos_trades", 0) for w in wfo_log)

        sep = "═" * 80
        logger.info("")
        logger.info(sep)
        logger.info("  📊 %s  RESUMEN WFO INSTITUCIONAL", self._prefix)
        logger.info(sep)
        logger.info("  Win | Ventana | IS Sharpe | OOS Ret % | OOS Sharpe | Trades | Params")
        logger.info("  " + "─" * 76)

        for w in wfo_log:
            is_win = "✅" if w["oos_return_pct"] > 0 else ("⚠️ " if w["oos_return_pct"] == 0 else "❌")
            params_short = (
                f"SL={w['best_params'].get('sl_atr_mult','?')} "
                f"TP={w['best_params'].get('tp_atr_mult','?')} "
                f"Conf={w['best_params'].get('min_confidence', w['best_params'].get('expiry_bars','?'))}"
            )
            logger.info(
                "  %s  W%02d   | %9.3f | %+8.2f%% | %10.3f | %6d | %s",
                is_win, w["window"], w["is_sharpe"], w.get("oos_return_pct", 0.0), w["oos_sharpe"],
                w.get("oos_trades", 0), params_short,
            )

        logger.info("  " + "─" * 76)
        logger.info("  📈 Retorno Acumulado OOS (Stitched):  %+.2f%%", total_return)
        logger.info("  📊 OOS Sharpe Promedio:               %.3f", avg_oos)
        logger.info("  🏆 Ventana Ganadora:  W%02d (Sharpe: %.3f)", best_window["window"], best_window["oos_sharpe"])
        logger.info("  ⚡ Ventana Perdedora: W%02d (Sharpe: %.3f)", worst_window["window"], worst_window["oos_sharpe"])
        logger.info("  📉 Win-Rate Ventanas: %.1f%% (%d positivas / %d negativas / %d neutras)",
                    win_rate, len(positive_oos), len(negative_oos), len(zero_oos))
        logger.info("  🔄 Trades OOS totales: %d", total_trades)
        logger.info("  ⏱️  Tiempo total WFO:  %.1f segundos", elapsed)
        logger.info(sep)
        logger.info("")


# ══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NEXUS Walk-Forward Optimizer v2")
    parser.add_argument(
        "--csv", type=str,
        default=str(os.path.join(os.path.dirname(__file__), "..", "data", "historical", "BTCUSDT_1h.csv")),
        help="Ruta al CSV histórico",
    )
    parser.add_argument("--is-days", type=float, default=90.0)
    parser.add_argument("--oos-days", type=float, default=30.0)
    parser.add_argument("--mode", type=str, default="spot", choices=["spot", "binary"])
    args = parser.parse_args()

    if os.path.exists(args.csv):
        wfo = WalkForwardOptimizer(
            args.csv, is_days=args.is_days, oos_days=args.oos_days,
            trading_mode=args.mode,
        )
        grid = [
            {"sl_atr_mult": 1.5, "tp_atr_mult": 2.5, "min_confidence": 0.60},
            {"sl_atr_mult": 2.0, "tp_atr_mult": 3.0, "min_confidence": 0.65},
        ]
        res = wfo.execute_wfo(grid)
        print(f"\n--- WFO Completed in {res['elapsed_seconds']}s ---")
        for log in res["wfo_log"]:
            print(f"W{log['window']:02d}: {log['best_params']} | IS={log['is_sharpe']:.3f} | OOS={log['oos_sharpe']:.3f} | Trades={log['oos_trades']}")
    else:
        print(f"[!] CSV no encontrado: {args.csv}")
