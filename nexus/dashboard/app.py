"""
NEXUS Trading System — Dashboard Backend (FastAPI)
==================================================
Servidor REST asíncrono estilo institucional para exponer telemétria 
y estado general de operaciones en tiempo real.
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="NEXUS Institutional Dashboard",
    description="Real-time Quantitative Trading Telemetry",
    version="1.0.0"
)

# CORS para permitir conexiones externas si es necesario
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOGS_FILE = os.path.join(BASE_DIR, "..", "logs", "telemetry.jsonl")

# Sirviendo el Frontend Vanilla
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Sirve el dashboard principal."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return "<h1>Dashboard UI not found. Please build frontend.</h1>"
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/v1/status")
async def get_system_status() -> Dict[str, Any]:
    """
    Retorna el estado general de NEXUS.
    (En un sistema real esto conectaría a la base de datos o Memcache)
    """
    return {
        "status": "ONLINE",
        "mode": "PAPER", # O LIVE, requeriría leer el state
        "uptime_hrs": 24.5,
        "capital": 10000.00,
        "open_positions": 2,
        "exposure_pct": 14.5,
        "drawdown_pct": 1.2
    }


@app.get("/api/v1/logs")
async def get_telemetry_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Lee las últimas 'limit' líneas del archivo JSONL de telemetría,
    exponiendo el reasoning de los agentes y trades ejecutados.
    """
    if not os.path.exists(LOGS_FILE):
        return []

    logs = []
    try:
        # Leemos el archivo y sacamos las ultimas N lineas (O(N) ingenuo pero suficiente para el tail)
        with open(LOGS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            recent_lines = lines[-limit:]
            for line in recent_lines:
                if line.strip():
                    logs.append(json.loads(line))
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Iniciamos el servidor en puerto 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
