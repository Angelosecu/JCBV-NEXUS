"""
NEXUS Trading System — Structured JSON Logger
=============================================
Exporta telemetría y razonamiento de Agentes en JSONLines (JSONL) para 
consumo asíncrono desde el Web Dashboard, o integración futura ELK/Splunk.
"""

import os
import json
import logging
from datetime import datetime
from threading import Lock
from typing import Dict, Any

class QuantLogger:
    """Implementa logging asíncrono/thread-safe a archivo .jsonl."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, log_path: str = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QuantLogger, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_path: str = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        # Por defecto guarda en la subcarpeta logs de nexus/
        if log_path is None:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
            os.makedirs(base_dir, exist_ok=True)
            self.log_path = os.path.join(base_dir, "telemetry.jsonl")
        else:
            self.log_path = log_path
            
        # Check/create empty file
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                pass
                
        self._write_lock = Lock()
        self._initialized = True
        
    def log_agent_decision(self, agent_name: str, symbol: str, confidence: float, decision: str, reasoning: str, metadata: Dict[str, Any] = None):
        """Registra una inferencia o decisión de un Agente (Bull/Bear/Árbitro)."""
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "agent_decision",
            "agent": agent_name,
            "symbol": symbol,
            "confidence": confidence,
            "decision": decision,
            "reasoning": reasoning,
            "metadata": metadata or {}
        }
        self._write_jsonl(payload)
        
    def log_trade_execution(self, symbol: str, action: str, size: float, price: float, trade_id: str):
        """Registra la ejecución real (Live o Paper) de un Trade."""
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "trade_execution",
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "trade_id": trade_id
        }
        self._write_jsonl(payload)
        
    def log_system_event(self, event_name: str, message: str, level: str = "INFO", data: Dict[str, Any] = None):
        """Registra eventos críticos del sistema (Errores, Init, Circuit Breakers)."""
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "system_event",
            "level": level.upper(),
            "event": event_name,
            "message": message,
            "data": data or {}
        }
        self._write_jsonl(payload)
        
    def _write_jsonl(self, payload: Dict[str, Any]):
        try:
            json_str = json.dumps(payload, ensure_ascii=False)
            with self._write_lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json_str + "\n")
        except Exception as e:
            logging.error(f"Error escribiendo JSONL en QuantLogger: {e}")

# Instancia global para fácil importación
get_quant_logger = lambda: QuantLogger()
