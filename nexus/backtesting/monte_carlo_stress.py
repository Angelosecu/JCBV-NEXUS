"""
NEXUS Trading System — Monte Carlo Stress Tester
===================================================
Implementa simulaciones estocásticas de Bootstraping con reemplazo para
comprobar el "Riesgo de Ruina" y el Riesgo de Secuencia de Retornos.
Aplica degradación artificial de retornos simulando stress macro.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger("nexus.monte_carlo")
if not logger.handlers:
    from config.settings import setup_logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

class MonteCarloStressTester:
    """Validador estocástico de ruina institucional."""

    def __init__(self, 
                 initial_capital: float = 10000.0, 
                 win_rate_degradation: float = 0.05, 
                 payoff_degradation: float = 0.10):
        """
        Args:
            initial_capital:      Capital inicial base para simulaciones.
            win_rate_degradation: Fricción artificial % aplicada al Win Rate original.
            payoff_degradation:   Fricción artificial % aplicada a ganancias brutas (Slippage/Fees/Alpha Decay).
        """
        self.initial_capital = initial_capital
        self.wr_deg = win_rate_degradation
        self.po_deg = payoff_degradation

    def run_simulation(self, trade_returns: List[float], iterations: int = 2000) -> Dict[str, Any]:
        """
        Ejecuta N iteraciones de Monte Carlo aplicando permutación y degradación.
        
        Args:
            trade_returns: Lista de retornos porcentuales históricos de los trades.
                           (e.g., [0.015, -0.01, 0.05, -0.005, ...])
            iterations:    Número de caminos simulados.
            
        Returns:
            Métricas de percentiles: P5 (Pesimista), P50(Medio), P95 (Optimista), y Ruin Risk.
        """
        if not trade_returns or len(trade_returns) < 10:
            logger.warning("Insuficientes retornos para Monte Carlo válido (>10).")
            return {"error": "Not enough data"}

        n_trades = len(trade_returns)
        arr = np.array(trade_returns, dtype=np.float64)
        
        # Simular degradación general 
        # Si return > 0, recortamos un X% (payoff_degradation)
        degraded_array = np.where(arr > 0, arr * (1.0 - self.po_deg), arr)
        
        # La degradación de Win Rate implica convertir % retornos aleatorios ganadores en perdedores
        # (Esto se simplificará restando el ratio general equivalente para todos, pero lo hacemos estocástico)
        win_indices = np.where(degraded_array > 0)[0]
        n_flips = int(len(win_indices) * self.wr_deg)
        
        final_results = []
        max_drawdowns = []
        ruin_events = 0
        ruin_threshold = self.initial_capital * 0.85 # 15% Max Drawdown Circuit Breaker
        
        logger.info("Iniciando Monte Carlo: %d Iteraciones | Degradación WR: %.1f%% | Degradación Payoff: %.1f%%", 
                    iterations, self.wr_deg * 100, self.po_deg * 100)

        for _ in range(iterations):
            # 1. Bootstrap sampling
            sim_returns = np.random.choice(degraded_array, size=n_trades, replace=True)
            
            # 2. Aplicar degradación de WR estocástica en esta muestra
            sim_wins = np.where(sim_returns > 0)[0]
            if len(sim_wins) > 0 and n_flips > 0:
                # Tomar aleatoriamente N sub-indices ganadores y volverlos cero o perdedores leves
                flips = np.random.choice(sim_wins, size=min(n_flips, len(sim_wins)), replace=False)
                sim_returns[flips] = -0.005 # Simular stop loss mínimo en esas fallas
            
            # 3. Reconstruir Curva
            equity = self.initial_capital
            peak = equity
            max_dd = 0.0
            ruined = False
            
            for r in sim_returns:
                equity *= (1 + r)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
                    
                if equity <= ruin_threshold and not ruined:
                    ruined = True
                    ruin_events += 1
                    
            final_results.append(equity)
            max_drawdowns.append(max_dd)

        final_arr = np.array(final_results)
        dd_arr = np.array(max_drawdowns)
        
        metrics = {
            "iterations": iterations,
            "ruin_probability_15pct": (ruin_events / iterations) * 100.0,
            
            "capital_p5": float(np.percentile(final_arr, 5)),
            "capital_p50": float(np.percentile(final_arr, 50)),
            "capital_p95": float(np.percentile(final_arr, 95)),
            
            "max_dd_p5": float(np.percentile(dd_arr, 5)) * 100,
            "max_dd_p50": float(np.percentile(dd_arr, 50)) * 100,
            "max_dd_p95": float(np.percentile(dd_arr, 95)) * 100,
        }
        
        logger.info("Monte Carlo Completado. Riesgo de Ruina (15%% CB): %.2f%%", metrics["ruin_probability_15pct"])
        logger.info("Expectativa de Peor Escenario (P5) DD: %.2f%% | Capital: $%.2f", metrics["max_dd_p95"], metrics["capital_p5"])
        
        return metrics

if __name__ == "__main__":
    # Test Unitario Simple
    tester = MonteCarloStressTester()
    # Retornos de ejemplo: 60% win rate promedio 1.5% gana / -1% pierde
    ejemplo = [0.015, 0.015, -0.01, 0.015, -0.01, 0.015, 0.015, -0.01, -0.01, 0.015] * 10
    
    res = tester.run_simulation(ejemplo, iterations=1000)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
