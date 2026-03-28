"""
NEXUS Trading System — Structured JSON Logger v2
===================================================
Telemetría Institucional Multi-Tier con rotación inteligente de archivos.

TIERS DE LOGGING:
  - CRITICAL (Nivel Crítico): Cache LLM, WFO, Crasheos fatales.
    → 10 MB max, 3 archivos rotatorios (30 MB total).
  - MEDIUM   (Nivel Medio):   Trading, rebalanceos, señales de agentes.
    → 10 MB max, 2 archivos rotatorios (20 MB total).
  - LOW      (Nivel Bajo):    Debug rutinario, heartbeats, tracing.
    → 5 MB max, 1 archivo rotatorio (5 MB total).

La purga de archivos viejos es responsabilidad del MaintenanceAssistant,
que evaluará timestamps antes de sobreescribir rotaciones en frío.
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
from threading import Lock
from typing import Dict, Any, Optional

# ══════════════════════════════════════════════════════════════════════
#  Constantes de Rotación
# ══════════════════════════════════════════════════════════════════════

_TZ_GMT5 = timezone(timedelta(hours=-5))
_BASE_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

# Tier definitions: (max_bytes, backup_count)
_TIER_CONFIG = {
    "critical": (10 * 1024 * 1024, 3),   # 10 MB x 3 = 30 MB
    "medium":   (10 * 1024 * 1024, 2),   # 10 MB x 2 = 20 MB
    "low":      (5  * 1024 * 1024, 1),   # 5 MB  x 1 = 5 MB
}

_TIER_FILES = {
    "critical": "nexus_critical.log",
    "medium":   "nexus_trading.log",
    "low":      "nexus_debug.log",
}


# ══════════════════════════════════════════════════════════════════════
#  QuantLogger v2 (Singleton, Thread-Safe, Tiered)
# ══════════════════════════════════════════════════════════════════════

class QuantLogger:
    """Logger estructurado JSONL con rotación de archivos por niveles."""

    _instance: Optional["QuantLogger"] = None
    _lock = Lock()

    def __new__(cls, log_dir: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QuantLogger, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir: Optional[str] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._log_dir = log_dir or _BASE_LOG_DIR
        os.makedirs(self._log_dir, exist_ok=True)

        # Crear handlers rotativos por tier
        self._handlers: Dict[str, RotatingFileHandler] = {}
        self._write_lock = Lock()

        _fmt = logging.Formatter("%(message)s")

        for tier, (max_bytes, backup_count) in _TIER_CONFIG.items():
            filepath = os.path.join(self._log_dir, _TIER_FILES[tier])
            handler = RotatingFileHandler(
                filepath,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            handler.setFormatter(_fmt)
            self._handlers[tier] = handler

        # JSONL legacy (telemetry.jsonl) — siempre se mantiene para el Dashboard
        self._jsonl_path = os.path.join(self._log_dir, "telemetry.jsonl")
        if not os.path.exists(self._jsonl_path):
            with open(self._jsonl_path, "w", encoding="utf-8") as f:
                pass

        self._initialized = True

    # ──────────────────────────────────────────────
    #  Tier-Aware Writing
    # ──────────────────────────────────────────────

    def _write_tier(self, tier: str, payload: Dict[str, Any]) -> None:
        """Escribe un payload JSON a un tier específico con rotación."""
        try:
            json_str = json.dumps(payload, ensure_ascii=False)
            record = logging.LogRecord(
                name="nexus", level=logging.INFO, pathname="",
                lineno=0, msg=json_str, args=None, exc_info=None,
            )
            handler = self._handlers.get(tier)
            if handler:
                with self._write_lock:
                    handler.emit(record)
        except Exception as e:
            logging.error("QuantLogger._write_tier error (%s): %s", tier, e)

    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        """Legacy JSONL append para el Dashboard en tiempo real."""
        try:
            json_str = json.dumps(payload, ensure_ascii=False)
            with self._write_lock:
                with open(self._jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json_str + "\n")
        except Exception as e:
            logging.error("QuantLogger._write_jsonl error: %s", e)

    def _timestamp(self) -> str:
        return datetime.now(_TZ_GMT5).isoformat()

    # ──────────────────────────────────────────────
    #  Public API — Agent Decisions (Tier MEDIUM)
    # ──────────────────────────────────────────────

    def log_agent_decision(
        self,
        agent_name: str,
        symbol: str,
        confidence: float,
        decision: str,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra una inferencia o decisión de un Agente (Bull/Bear/Árbitro)."""
        payload = {
            "timestamp": self._timestamp(),
            "type": "agent_decision",
            "agent": agent_name,
            "symbol": symbol,
            "confidence": confidence,
            "decision": decision,
            "reasoning": reasoning,
            "metadata": metadata or {},
        }
        self._write_tier("medium", payload)
        self._write_jsonl(payload)

    # ──────────────────────────────────────────────
    #  Public API — Trade Execution (Tier MEDIUM)
    # ──────────────────────────────────────────────

    def log_trade_execution(
        self,
        symbol: str,
        action: str,
        size: float,
        price: float,
        trade_id: str,
    ) -> None:
        """Registra la ejecución real (Live o Paper) de un Trade."""
        payload = {
            "timestamp": self._timestamp(),
            "type": "trade_execution",
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "trade_id": trade_id,
        }
        self._write_tier("medium", payload)
        self._write_jsonl(payload)

    # ──────────────────────────────────────────────
    #  Public API — System Events (Tier by severity)
    # ──────────────────────────────────────────────

    def log_system_event(
        self,
        event_name: str,
        message: str,
        level: str = "INFO",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra eventos críticos del sistema (Errores, Init, Circuit Breakers)."""
        payload = {
            "timestamp": self._timestamp(),
            "type": "system_event",
            "level": level.upper(),
            "event": event_name,
            "message": message,
            "data": data or {},
        }
        # Route to tier based on severity
        if level.upper() in ("CRITICAL", "ERROR", "FATAL"):
            tier = "critical"
        elif level.upper() in ("WARNING", "INFO"):
            tier = "medium"
        else:
            tier = "low"

        self._write_tier(tier, payload)
        self._write_jsonl(payload)

    # ──────────────────────────────────────────────
    #  Public API — Crash & Diagnostic (Tier CRITICAL)
    # ──────────────────────────────────────────────

    def log_crash(
        self,
        module: str,
        error_type: str,
        traceback_str: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra un crasheo fatal con StackTrace completo."""
        payload = {
            "timestamp": self._timestamp(),
            "type": "crash_report",
            "module": module,
            "error_type": error_type,
            "traceback": traceback_str,
            "context": context or {},
        }
        self._write_tier("critical", payload)
        self._write_jsonl(payload)

    # ──────────────────────────────────────────────
    #  Public API — Maintenance Events (Tier LOW)
    # ──────────────────────────────────────────────

    def log_maintenance(
        self,
        action: str,
        details: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra acciones de mantenimiento automático (purgas, rotaciones)."""
        payload = {
            "timestamp": self._timestamp(),
            "type": "maintenance",
            "action": action,
            "details": details,
            "data": data or {},
        }
        self._write_tier("low", payload)

    # ──────────────────────────────────────────────
    #  Public API — WFO/Calibration (Tier CRITICAL)
    # ──────────────────────────────────────────────

    def log_calibration(
        self,
        mode: str,
        symbol: str,
        window: int,
        params: Dict[str, Any],
        sharpe: float,
        phase: str = "IS",
    ) -> None:
        """Registra progreso de calibración WFO."""
        payload = {
            "timestamp": self._timestamp(),
            "type": "wfo_calibration",
            "mode": mode,
            "symbol": symbol,
            "window": window,
            "phase": phase,
            "params": params,
            "sharpe": sharpe,
        }
        self._write_tier("critical", payload)

    # ──────────────────────────────────────────────
    #  Tier Stats (para el Maintenance Assistant)
    # ──────────────────────────────────────────────

    def get_tier_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retorna tamaños actuales de cada tier para el Maintenance Assistant."""
        stats = {}
        for tier, filename in _TIER_FILES.items():
            filepath = os.path.join(self._log_dir, filename)
            total_bytes = 0
            file_count = 0

            # Archivo principal
            if os.path.exists(filepath):
                total_bytes += os.path.getsize(filepath)
                file_count += 1

            # Archivos rotados (.1, .2, .3...)
            max_backups = _TIER_CONFIG[tier][1]
            for i in range(1, max_backups + 1):
                rotated = f"{filepath}.{i}"
                if os.path.exists(rotated):
                    total_bytes += os.path.getsize(rotated)
                    file_count += 1

            max_bytes = _TIER_CONFIG[tier][0]
            max_total = max_bytes * (_TIER_CONFIG[tier][1] + 1)

            stats[tier] = {
                "current_bytes": total_bytes,
                "max_total_bytes": max_total,
                "usage_pct": round((total_bytes / max_total) * 100, 1) if max_total > 0 else 0,
                "file_count": file_count,
                "path": filepath,
            }
        return stats


# ══════════════════════════════════════════════════════════════════════
#  Factory Global
# ══════════════════════════════════════════════════════════════════════

def get_quant_logger() -> QuantLogger:
    """Instancia global singleton para fácil importación."""
    return QuantLogger()
