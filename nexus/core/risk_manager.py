"""
NEXUS Trading System — Quant Risk Manager
============================================
Gestión cuantitativa de riesgo con verificación matemática inline.

5 métodos principales:
  1. kelly_criterion()          — Dimensionamiento óptimo (Kelly fraccionado)
  2. monte_carlo_simulation()   — Trayectorias de capital aleatorias
  3. value_at_risk()            — VaR paramétrico (μ - z·σ)
  4. circuit_breaker_check()    — Corte de emergencia por drawdown
  5. correlation_check()        — Bloqueo por correlación excesiva

Cada método incluye un bloque de validación assert al final del módulo.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

try:
    from scipy import stats as scipy_stats  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger("nexus.risk_manager")


# ══════════════════════════════════════════════════════════════════════
#  Data Classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PositionSize:
    """Resultado del cálculo de tamaño de posición."""
    optimal_size: float
    kelly_size: float
    adjusted_size: float
    max_allowed: float
    rationale: str


@dataclass
class RiskMetrics:
    """Métricas de riesgo del portafolio."""
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    current_exposure: float


@dataclass
class MonteCarloResult:
    """Resultado de la simulación Monte Carlo."""
    p5: float               # Percentil 5
    p50: float              # Mediana
    p95: float              # Percentil 95
    max_dd: float           # Drawdown máximo promedio
    simulations: int
    paths: Optional[np.ndarray] = None


# ══════════════════════════════════════════════════════════════════════
#  QuantRiskManager
# ══════════════════════════════════════════════════════════════════════

class QuantRiskManager:
    """
    Gestor cuantitativo de riesgo principal.

    Integra Kelly Criterion, Monte Carlo, VaR paramétrico,
    circuit breaker y control de correlación.
    """

    # Constantes
    KELLY_MAX_FRACTION = 0.25           # Cap del Kelly fraccionado
    Z_SCORES = {0.95: 1.645, 0.99: 2.326}
    CIRCUIT_BREAKER_COOLDOWN = 86_400   # 24 horas en segundos

    def __init__(
        self,
        log_dir: str = "logs",
        execution_engine: Any = None,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._execution_engine = execution_engine
        self._portfolio_value: float = 0.0
        self._open_positions: List[Dict[str, Any]] = []
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_until: float = 0.0  # timestamp

    # ══════════════════════════════════════════════════════════════════
    #  MÉTODO 1: Kelly Criterion
    # ══════════════════════════════════════════════════════════════════

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calcula la fracción óptima de Kelly.

        Fórmula: f* = (b·p − q) / b
        donde:
            b = avg_win / avg_loss  (ratio ganancia/pérdida)
            p = win_rate            (probabilidad de ganar)
            q = 1 − win_rate        (probabilidad de perder)

        Reglas:
            - Si f* > 0.25 → retornar 0.25 (Kelly fraccionado)
            - Si f* < 0    → retornar 0.0  (no operar)

        Args:
            win_rate: Probabilidad de trade ganador (0-1)
            avg_win:  Ganancia promedio por trade ganador (valor absoluto)
            avg_loss: Pérdida promedio por trade perdedor (valor absoluto)

        Returns:
            Fracción óptima del capital a apostar (0.0 – 0.25)
        """
        if avg_loss <= 0:
            raise ValueError(f"avg_loss debe ser > 0, recibido: {avg_loss}")
        if not (0 <= win_rate <= 1):
            raise ValueError(f"win_rate debe estar entre 0 y 1, recibido: {win_rate}")

        b = avg_win / avg_loss          # Win/Loss ratio
        p = win_rate                    # Prob. de ganar
        q = 1.0 - win_rate              # Prob. de perder

        f_star = (b * p - q) / b        # Kelly puro

        # Regla: si negativo → no operar
        if f_star < 0:
            logger.info("Kelly f*=%.4f < 0 → no operar", f_star)
            return 0.0

        # Regla: cap a 0.25 (Kelly fraccionado)
        result = min(f_star, self.KELLY_MAX_FRACTION)

        logger.info(
            "Kelly: p=%.2f, b=%.2f → f*=%.4f → cappado=%.4f",
            p, b, f_star, result,
        )
        return round(result, 6)  # type: ignore

    # ══════════════════════════════════════════════════════════════════
    #  MÉTODO 2: Monte Carlo Simulation
    # ══════════════════════════════════════════════════════════════════

    def monte_carlo_simulation(
        self,
        returns_list: List[float],
        n: int = 10_000,
        initial_capital: float = 10_000.0,
        horizon: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Genera n trayectorias de capital aleatorias usando remuestreo
        con reemplazo de retornos históricos.

        Args:
            returns_list: Lista de retornos porcentuales (e.g. 0.01 = +1%)
            n:            Número de simulaciones
            initial_capital: Capital inicial
            horizon:      Número de pasos por trayectoria (default = len(returns_list))

        Returns:
            {"p5": float, "p50": float, "p95": float, "max_dd": float}
        """
        if not returns_list:
            raise ValueError("returns_list no puede estar vacío")

        returns_arr = np.array(returns_list, dtype=np.float64)
        h = horizon or len(returns_arr)

        # Remuestreo con reemplazo: (n, h) samples
        rng = np.random.default_rng(seed=42)
        sampled = rng.choice(returns_arr, size=(n, h), replace=True)

        # Trayectorias de capital acumuladas
        # capital[i, t] = capital_0 * prod(1 + r[j] para j=0..t)
        cumulative = initial_capital * np.cumprod(1.0 + sampled, axis=1)

        # Capital final de cada trayectoria
        final_capitals = cumulative[:, -1]  # type: ignore

        # Percentiles del capital final
        p5 = float(np.percentile(final_capitals, 5))
        p50 = float(np.percentile(final_capitals, 50))
        p95 = float(np.percentile(final_capitals, 95))

        # Drawdown máximo promedio
        # Para cada trayectoria, calcular max drawdown
        running_max = np.maximum.accumulate(cumulative, axis=1)
        drawdowns = (running_max - cumulative) / running_max
        max_dds = np.max(drawdowns, axis=1)
        avg_max_dd = float(np.mean(max_dds))

        result = {
            "p5": round(p5, 2),  # type: ignore
            "p50": round(p50, 2),  # type: ignore
            "p95": round(p95, 2),  # type: ignore
            "max_dd": round(avg_max_dd, 6),  # type: ignore
        }

        logger.info(
            "Monte Carlo (%d sims, h=%d): p5=%.2f, p50=%.2f, p95=%.2f, max_dd=%.4f",
            n, h, p5, p50, p95, avg_max_dd,
        )
        return result

    # ══════════════════════════════════════════════════════════════════
    #  MÉTODO 3: Value at Risk (VaR paramétrico)
    # ══════════════════════════════════════════════════════════════════

    def value_at_risk(
        self,
        returns_list: List[float],
        confidence: float = 0.95,
    ) -> float:
        """
        VaR paramétrico (distribución normal).

        Fórmula: VaR = μ − z·σ
        donde z = 1.645 para 95%, z = 2.326 para 99%

        Args:
            returns_list: Lista de retornos porcentuales
            confidence:   Nivel de confianza (0.95 o 0.99)

        Returns:
            VaR como porcentaje del capital en riesgo (valor positivo = pérdida)
        """
        if not returns_list:
            raise ValueError("returns_list no puede estar vacío")

        returns_arr = np.array(returns_list, dtype=np.float64)
        mu = float(np.mean(returns_arr))
        sigma = float(np.std(returns_arr, ddof=1))  # Desviación estándar muestral

        z = self.Z_SCORES.get(confidence)
        if z is None:
            # Calcular z-score genérico
            if _HAS_SCIPY:
                z = scipy_stats.norm.ppf(confidence)
            else:
                raise ValueError(
                    f"confidence={confidence} no soportado sin scipy. "
                    f"Valores soportados: {list(self.Z_SCORES.keys())}"
                )

        # VaR = μ − z·σ  (negativo significa pérdida)
        var_value = mu - z * sigma

        # Retornar como valor positivo (porcentaje de capital en riesgo)
        var_pct = abs(var_value) if var_value < 0 else 0.0

        logger.info(
            "VaR(%.0f%%): μ=%.6f, σ=%.6f, z=%.3f → VaR=%.6f (%.4f%%)",
            confidence * 100, mu, sigma, z, var_value, var_pct * 100,
        )
        return round(var_pct, 6)  # type: ignore

    # ══════════════════════════════════════════════════════════════════
    #  MÉTODO 4: Circuit Breaker
    # ══════════════════════════════════════════════════════════════════

    def circuit_breaker_check(
        self,
        current_drawdown: float,
        max_dd: float = 0.15,
    ) -> bool:
        """
        Verifica si el drawdown actual excede el límite y activa
        el circuit breaker si es necesario.

        Si current_drawdown >= max_dd:
            1. Llama execution_engine.close_all_positions()
            2. Escribe en logs/circuit_breaker.log con timestamp
            3. Bloquea nuevas órdenes durante 86400 segundos (24h)
            4. Retorna True (activado)

        Args:
            current_drawdown: Drawdown actual como fracción (e.g. 0.18 = 18%)
            max_dd:           Límite máximo de drawdown (default: 0.15 = 15%)

        Returns:
            True si circuit breaker fue activado, False si no
        """
        if current_drawdown < max_dd:
            # Verificar si estamos en cooldown activo
            if self._circuit_breaker_active:
                remaining = self._circuit_breaker_until - time.time()
                if remaining > 0:
                    logger.warning(
                        "Circuit breaker activo (%.0f seg restantes)", remaining
                    )
                    return True
                else:
                    # Cooldown expirado
                    self._circuit_breaker_active = False
                    logger.info("Circuit breaker desactivado (cooldown expirado)")
            return False

        # ── Activar circuit breaker ───────────────────────────────────

        now = datetime.now(timezone.utc)
        self._circuit_breaker_active = True
        self._circuit_breaker_until = time.time() + self.CIRCUIT_BREAKER_COOLDOWN

        # 1. Cerrar todas las posiciones
        if self._execution_engine is not None:
            try:
                self._execution_engine.close_all_positions()
                logger.critical(
                    "CIRCUIT BREAKER: Todas las posiciones cerradas (DD=%.1f%% >= %.1f%%)",
                    current_drawdown * 100, max_dd * 100,
                )
            except Exception as exc:
                logger.error("Error cerrando posiciones: %s", exc)
        else:
            logger.warning("execution_engine no configurado — no se cerraron posiciones")

        # 2. Escribir en log
        log_path = self._log_dir / "circuit_breaker.log"
        log_entry = (
            f"[{now.isoformat()}] CIRCUIT BREAKER ACTIVADO | "
            f"drawdown={current_drawdown:.4f} ({current_drawdown:.1%}) | "
            f"max_dd={max_dd:.4f} ({max_dd:.1%}) | "
            f"cooldown={self.CIRCUIT_BREAKER_COOLDOWN}s\n"
        )
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
            logger.info("Circuit breaker log escrito en %s", log_path)
        except OSError as exc:
            logger.error("Error escribiendo circuit breaker log: %s", exc)

        # 3. Bloqueo ya está activo (self._circuit_breaker_active = True)
        logger.critical(
            "CIRCUIT BREAKER: Nuevas órdenes bloqueadas por %d segundos",
            self.CIRCUIT_BREAKER_COOLDOWN,
        )

        return True

    def is_circuit_breaker_active(self) -> bool:
        """Retorna True si el circuit breaker está activo y en cooldown."""
        if not self._circuit_breaker_active:
            return False
        if time.time() >= self._circuit_breaker_until:
            self._circuit_breaker_active = False
            return False
        return True

    # ══════════════════════════════════════════════════════════════════
    #  MÉTODO 5: Correlation Check
    # ══════════════════════════════════════════════════════════════════

    def correlation_check(
        self,
        open_positions: List[Dict[str, Any]],
        threshold: float = 0.85,
    ) -> bool:
        """
        Verifica la correlación entre posiciones abiertas.

        Usa scipy.stats.pearsonr entre los retornos de cada par.
        Si cualquier par tiene correlación > threshold → retorna False (bloquear).

        Args:
            open_positions: Lista de dicts con key "returns" (List[float])
            threshold:      Umbral de correlación para bloquear (default 0.85)

        Returns:
            True  = posiciones permitidas (sin correlación excesiva)
            False = bloquear (correlación > threshold detectada)
        """
        if not _HAS_SCIPY:
            logger.warning("scipy no disponible — correlation_check deshabilitado")
            return True

        if len(open_positions) < 2:
            return True  # Nada que comparar

        # Extraer retornos de cada posición
        returns_lists: List[np.ndarray] = []
        symbols: List[str] = []
        for pos in open_positions:
            rets = pos.get("returns", [])
            sym = pos.get("symbol", "???")
            if len(rets) >= 2:
                returns_lists.append(np.array(rets, dtype=np.float64))
                symbols.append(sym)

        if len(returns_lists) < 2:
            return True

        # Comparar todos los pares
        for i in range(len(returns_lists)):
            for j in range(i + 1, len(returns_lists)):
                # Igualar longitudes
                min_len = min(len(returns_lists[i]), len(returns_lists[j]))  # type: ignore
                if min_len < 2:
                    continue

                r1 = returns_lists[i][:min_len]  # type: ignore
                r2 = returns_lists[j][:min_len]  # type: ignore

                corr, p_value = scipy_stats.pearsonr(r1, r2)

                logger.debug(
                    "Correlación %s/%s: r=%.4f, p=%.4f",
                    symbols[i], symbols[j], corr, p_value,  # type: ignore
                )

                if abs(corr) > threshold:  # type: ignore
                    logger.warning(
                        "BLOQUEO: Correlación excesiva entre %s y %s "
                        "(r=%.4f > %.2f)",
                        symbols[i], symbols[j], corr, threshold,  # type: ignore
                    )
                    return False

        logger.info("Correlation check OK: todas las posiciones bajo umbral %.2f", threshold)
        return True

    # ══════════════════════════════════════════════════════════════════
    #  Métodos auxiliares para el sistema principal
    # ══════════════════════════════════════════════════════════════════

    def update_portfolio(
        self, portfolio_value: float, positions: List[Dict[str, Any]]
    ) -> None:
        """Actualiza el estado del portafolio."""
        self._portfolio_value = portfolio_value
        self._open_positions = positions

    def get_risk_report(self) -> Dict[str, Any]:
        """Genera un reporte resumido de riesgo para el árbitro."""
        return {
            "portfolio_value": self._portfolio_value,
            "num_positions": len(self._open_positions),
            "circuit_breaker_active": self.is_circuit_breaker_active(),
        }

    def __repr__(self) -> str:
        status = "CB_ACTIVE" if self.is_circuit_breaker_active() else "OK"
        return (
            f"<QuantRiskManager status={status} "
            f"positions={len(self._open_positions)}>"
        )


# ══════════════════════════════════════════════════════════════════════
#  BLOQUE DE VALIDACIÓN — Asserts con valores de prueba conocidos
# ══════════════════════════════════════════════════════════════════════

def _run_validation() -> bool:
    """
    Ejecuta todos los asserts de verificación matemática.
    Retorna True si todos pasan, False si alguno falla.
    """
    import sys
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore

    rm = QuantRiskManager(log_dir="logs")
    passed = 0
    failed = 0
    total = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed, total
        total += 1  # type: ignore
        if condition:
            passed += 1  # type: ignore
            print(f"  [OK]  {name}")
        else:
            failed += 1  # type: ignore
            print(f"  [FAIL] {name}: {detail}")

    print("\n" + "=" * 60)
    print("  VALIDACION MATEMATICA — QuantRiskManager")
    print("=" * 60)

    # ── KELLY CRITERION ───────────────────────────────────────────

    print("\n--- Kelly Criterion ---")

    # Test 1: kelly(0.55, 2.0, 1.0) → f* = (2*0.55 - 0.45)/2 = 0.325 → cap a 0.25
    k1 = rm.kelly_criterion(0.55, 2.0, 1.0)
    f_star_1 = (2.0 * 0.55 - 0.45) / 2.0  # = 0.325
    check(
        "kelly(0.55, 2.0, 1.0) = 0.25 (cappado)",
        k1 == 0.25,
        f"Esperado 0.25 (f*={f_star_1:.4f}), obtenido {k1}",
    )

    # Test 2: kelly(0.50, 1.0, 1.0) → f* = (1*0.5 - 0.5)/1 = 0.0
    k2 = rm.kelly_criterion(0.50, 1.0, 1.0)
    check(
        "kelly(0.50, 1.0, 1.0) = 0.0 (breakeven)",
        k2 == 0.0,
        f"Esperado 0.0, obtenido {k2}",
    )

    # Test 3: kelly(0.40, 1.0, 1.0) → f* = (1*0.4 - 0.6)/1 = -0.2 → 0.0
    k3 = rm.kelly_criterion(0.40, 1.0, 1.0)
    check(
        "kelly(0.40, 1.0, 1.0) = 0.0 (negativo)",
        k3 == 0.0,
        f"Esperado 0.0, obtenido {k3}",
    )

    # Test 4: kelly(0.60, 1.5, 1.0) → f* = (1.5*0.6 - 0.4)/1.5 = 0.3333 → cap 0.25
    k4 = rm.kelly_criterion(0.60, 1.5, 1.0)
    check(
        "kelly(0.60, 1.5, 1.0) = 0.25 (cappado)",
        k4 == 0.25,
        f"Esperado 0.25, obtenido {k4}",
    )

    # Test 5: kelly(0.55, 1.2, 1.0) → f* = (1.2*0.55 - 0.45)/1.2 = 0.175
    k5 = rm.kelly_criterion(0.55, 1.2, 1.0)
    expected_k5 = round((1.2 * 0.55 - 0.45) / 1.2, 6)  # type: ignore
    check(
        f"kelly(0.55, 1.2, 1.0) = {expected_k5} (sin cap)",
        k5 == expected_k5,
        f"Esperado {expected_k5}, obtenido {k5}",
    )

    # ── MONTE CARLO ───────────────────────────────────────────────

    print("\n--- Monte Carlo Simulation ---")

    # Test 1: retornos constantes de +1% → p50 debe ser positivo
    constant_returns = [0.01] * 30
    mc1 = rm.monte_carlo_simulation(constant_returns, n=1000)
    check(
        "MC retornos +1% constante → p50 > capital inicial",
        mc1["p50"] > 10_000.0,
        f"p50={mc1['p50']}, esperado > 10000",
    )
    check(
        "MC retornos +1% → p5 > capital inicial",
        mc1["p5"] > 10_000.0,
        f"p5={mc1['p5']}",
    )
    check(
        "MC retornos +1% → max_dd = 0 (sin drawdowns)",
        mc1["max_dd"] == 0.0,
        f"max_dd={mc1['max_dd']}",
    )

    # Test 2: retornos mixtos → p5 < p50 < p95
    mixed_returns = [0.02, -0.01, 0.015, -0.005, 0.01, -0.02, 0.03, -0.01]
    mc2 = rm.monte_carlo_simulation(mixed_returns, n=5000)
    check(
        "MC retornos mixtos → p5 < p50 < p95",
        mc2["p5"] < mc2["p50"] < mc2["p95"],
        f"p5={mc2['p5']}, p50={mc2['p50']}, p95={mc2['p95']}",
    )
    check(
        "MC retornos mixtos → max_dd > 0",
        mc2["max_dd"] > 0,
        f"max_dd={mc2['max_dd']}",
    )

    # Test 3: retornos negativos constantes → p50 < capital inicial
    neg_returns = [-0.02] * 20
    mc3 = rm.monte_carlo_simulation(neg_returns, n=1000)
    check(
        "MC retornos -2% constante → p50 < capital inicial",
        mc3["p50"] < 10_000.0,
        f"p50={mc3['p50']}",
    )

    # ── VALUE AT RISK ─────────────────────────────────────────────

    print("\n--- Value at Risk (VaR parametrico) ---")

    # Test 1: retornos conocidos → verificar fórmula
    var_returns = [0.01, 0.02, -0.01, 0.005, -0.015, 0.03, -0.005, 0.01, -0.02, 0.015]
    mu = np.mean(var_returns)
    sigma = np.std(var_returns, ddof=1)
    expected_var = abs(mu - 1.645 * sigma) if (mu - 1.645 * sigma) < 0 else 0
    var1 = rm.value_at_risk(var_returns, confidence=0.95)
    check(
        "VaR(95%) fórmula = |μ - 1.645·σ|",
        abs(var1 - round(expected_var, 6)) < 1e-5,
        f"Esperado {expected_var:.6f}, obtenido {var1}",
    )

    # Test 2: retornos todos positivos → VaR debería ser 0 o muy bajo
    pos_returns = [0.01, 0.02, 0.015, 0.025, 0.01]
    var2 = rm.value_at_risk(pos_returns, confidence=0.95)
    check(
        "VaR con retornos positivos = 0 (μ - z·σ > 0)",
        var2 == 0.0 or var2 < 0.01,
        f"VaR={var2}",
    )

    # ── CIRCUIT BREAKER ───────────────────────────────────────────

    print("\n--- Circuit Breaker ---")

    # Test 1: DD bajo → no activar
    rm_cb = QuantRiskManager(log_dir="logs")
    cb1 = rm_cb.circuit_breaker_check(0.05, max_dd=0.15)
    check(
        "CB drawdown 5% < 15% → False (no activar)",
        cb1 is False,
        f"Obtenido {cb1}",
    )

    # Test 2: DD alto → activar
    cb2 = rm_cb.circuit_breaker_check(0.18, max_dd=0.15)
    check(
        "CB drawdown 18% >= 15% → True (activar)",
        cb2 is True,
        f"Obtenido {cb2}",
    )

    # Test 3: Verificar que circuit breaker quedó activo
    check(
        "CB activo después de activación",
        rm_cb.is_circuit_breaker_active() is True,
        f"Obtenido {rm_cb.is_circuit_breaker_active()}",
    )

    # Test 4: DD bajo pero CB activo → sigue activo
    cb3 = rm_cb.circuit_breaker_check(0.02, max_dd=0.15)
    check(
        "CB activo: DD bajo pero en cooldown → True",
        cb3 is True,
        f"Obtenido {cb3}",
    )

    # ── CORRELATION CHECK ─────────────────────────────────────────

    print("\n--- Correlation Check ---")

    if _HAS_SCIPY:
        # Test 1: Posiciones idénticas → correlación 1.0 → bloquear
        pos_identical = [
            {"symbol": "BTC", "returns": [0.01, 0.02, -0.01, 0.015, -0.005]},
            {"symbol": "BTC2", "returns": [0.01, 0.02, -0.01, 0.015, -0.005]},
        ]
        cc1 = rm.correlation_check(pos_identical, threshold=0.85)
        check(
            "Correlación idéntica (r=1.0) > 0.85 → False (bloquear)",
            cc1 is False,
            f"Obtenido {cc1}",
        )

        # Test 2: Posiciones no correlacionadas → permitir
        pos_uncorr = [
            {"symbol": "BTC", "returns": [0.01, -0.02, 0.03, -0.01, 0.02, -0.005, 0.015]},
            {"symbol": "GOLD", "returns": [0.005, 0.01, -0.005, 0.02, -0.01, 0.015, -0.002]},
        ]
        cc2 = rm.correlation_check(pos_uncorr, threshold=0.85)
        check(
            "Correlación baja → True (permitir)",
            cc2 is True,
            f"Obtenido {cc2}",
        )

        # Test 3: Una sola posición → siempre permitir
        pos_single = [{"symbol": "BTC", "returns": [0.01, 0.02]}]
        cc3 = rm.correlation_check(pos_single, threshold=0.85)
        check(
            "Una sola posición → True (nada que comparar)",
            cc3 is True,
            f"Obtenido {cc3}",
        )
    else:
        print("  [SKIP] scipy no instalado — correlation checks omitidos")

    # ── RESULTADO FINAL ───────────────────────────────────────────

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  RISK MANAGER VALIDADO: {passed}/{total} tests passed")
    else:
        print(f"  ERRORES ENCONTRADOS: {failed}/{total} tests fallaron")
    print("=" * 60)

    return failed == 0


# Ejecutar validación cuando se corre el archivo directamente
if __name__ == "__main__":
    import sys
    success = _run_validation()
    sys.exit(0 if success else 1)
