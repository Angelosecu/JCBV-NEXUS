"""
NEXUS Trading System — Data Handler
=====================================
Conexión WebSocket a Binance Futures, ingestión y limpieza de datos
de mercado en tiempo real, almacenamiento en SQLite vía SQLAlchemy.

Usa python-binance AsyncClient para operaciones no bloqueantes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict, deque

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sqlalchemy import (  # type: ignore
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker  # type: ignore

from binance import AsyncClient, BinanceSocketManager  # type: ignore
from binance.enums import HistoricalKlinesType  # type: ignore

from config.settings import (  # type: ignore
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    DATABASE_URL,
    trading_config,
)

logger = logging.getLogger("nexus.data_handler")

# ──────────────────────────────────────────────────────────────────────
#  SQLAlchemy Model
# ──────────────────────────────────────────────────────────────────────

Base = declarative_base()


class KlineRecord(Base):
    """Modelo SQLAlchemy para almacenar velas OHLCV."""

    __tablename__ = "klines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    interval = Column(String(5), nullable=False, index=True)
    open_time = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    close_time = Column(DateTime, nullable=False)
    quote_volume = Column(Float, nullable=False)
    num_trades = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time", name="uq_kline"),
    )


# ──────────────────────────────────────────────────────────────────────
#  DataStore — SQLite persistence
# ──────────────────────────────────────────────────────────────────────


class DataStore:
    """Almacenamiento persistente de datos de mercado con SQLAlchemy."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        self.db_url = db_url or DATABASE_URL
        self._engine = None
        self._SessionFactory = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Crea engine, session factory y tablas si no existen."""
        # Configuración específica para SQLite
        connect_args = {}
        if self.db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}

        self._engine = create_engine(
            self.db_url,
            echo=False,
            connect_args=connect_args,
        )
        Base.metadata.create_all(self._engine)
        self._SessionFactory = sessionmaker(bind=self._engine)
        logger.info("DataStore inicializado → %s", self.db_url)

    def _session(self) -> Session:
        return self._SessionFactory()  # type: ignore

    # ── write ─────────────────────────────────────────────────────────

    def save_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        """Persiste un DataFrame de velas.  Retorna filas insertadas."""
        if df.empty:
            return 0

        session = self._session()
        inserted = 0
        try:
            for _, row in df.iterrows():
                # Upsert — ignorar duplicados
                exists = (
                    session.query(KlineRecord)
                    .filter_by(
                        symbol=symbol,
                        interval=interval,
                        open_time=row["open_time"],
                    )
                    .first()
                )
                if exists:
                    continue

                record = KlineRecord(
                    symbol=symbol,  # type: ignore
                    interval=interval,  # type: ignore
                    open_time=row["open_time"],  # type: ignore
                    open=float(row["open"]),  # type: ignore
                    high=float(row["high"]),  # type: ignore
                    low=float(row["low"]),  # type: ignore
                    close=float(row["close"]),  # type: ignore
                    volume=float(row["volume"]),  # type: ignore
                    close_time=row["close_time"],  # type: ignore
                    quote_volume=float(row.get("quote_volume", 0)),  # type: ignore
                    num_trades=int(row.get("num_trades", 0)),  # type: ignore
                )
                session.add(record)
                inserted += 1
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        logger.debug("Guardadas %d velas [%s %s]", inserted, symbol, interval)
        return inserted

    # ── read ──────────────────────────────────────────────────────────

    def load_klines(
        self,
        symbol: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Carga velas históricas filtradas por rango de fechas."""
        session = self._session()
        try:
            query = session.query(KlineRecord).filter_by(
                symbol=symbol, interval=interval
            )
            if start:
                query = query.filter(KlineRecord.open_time >= start)
            if end:
                query = query.filter(KlineRecord.open_time <= end)
            query = query.order_by(KlineRecord.open_time)

            records = query.all()
            if not records:
                return pd.DataFrame()

            data = [
                {
                    "open_time": r.open_time,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "close_time": r.close_time,
                    "quote_volume": r.quote_volume,
                    "num_trades": r.num_trades,
                }
                for r in records
            ]
            df = pd.DataFrame(data)
            if not df.empty:
                df["open_time"] = pd.to_datetime(df["open_time"]).dt.tz_localize("UTC")
                df["close_time"] = pd.to_datetime(df["close_time"]).dt.tz_localize("UTC")
            return df
        finally:
            session.close()

    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Retorna el open_time más reciente almacenado para un par/intervalo."""
        session = self._session()
        try:
            record = (
                session.query(KlineRecord)
                .filter_by(symbol=symbol, interval=interval)
                .order_by(KlineRecord.open_time.desc())
                .first()
            )
            return record.open_time if record else None
        finally:
            session.close()


# ──────────────────────────────────────────────────────────────────────
#  DataCleaner — limpieza y normalización
# ──────────────────────────────────────────────────────────────────────


class DataCleaner:
    """Limpieza, normalización y detección de outliers para datos OHLCV."""

    @staticmethod
    def clean_kline_row(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Limpia y valida una vela cruda tal como llega del WebSocket."""
        return {
            "open_time": datetime.fromtimestamp(raw["t"] / 1000, tz=timezone.utc),
            "open": float(raw["o"]),
            "high": float(raw["h"]),
            "low": float(raw["l"]),
            "close": float(raw["c"]),
            "volume": float(raw["v"]),
            "close_time": datetime.fromtimestamp(raw["T"] / 1000, tz=timezone.utc),
            "quote_volume": float(raw.get("q", 0)),
            "num_trades": int(raw.get("n", 0)),
            "is_closed": raw.get("x", False),
        }

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        column: str = "close",
        z_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Marca outliers basándose en z-score.  Añade columna 'is_outlier'."""
        if df.empty or column not in df.columns:
            return df

        df = df.copy()
        mean = df[column].mean()
        std = df[column].std()
        if std == 0:
            df["is_outlier"] = False
            return df

        df["z_score"] = (df[column] - mean) / std
        df["is_outlier"] = df["z_score"].abs() > z_threshold
        outlier_count = df["is_outlier"].sum()
        if outlier_count:
            logger.warning(
                "Detectados %d outliers en columna '%s' (z > %.1f)",
                outlier_count,
                column,
                z_threshold,
            )
        return df

    @staticmethod
    def fill_gaps(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Rellena huecos temporales con forward-fill."""
        if df.empty:
            return df
        df = df.copy()
        df = df.set_index("open_time")
        df = df.asfreq(freq)
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].ffill()
        df["volume"] = df["volume"].fillna(0)
        df = df.reset_index()
        return df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de limpieza:
        1. Elimina filas con NaN en OHLCV
        2. Convierte tipos numéricos
        3. Detecta y filtra outliers
        4. Ordena por timestamp
        """
        if df.empty:
            return df

        df = df.copy()

        # 1. Asegurar tipos numéricos
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2. Eliminar filas con NaN en columnas OHLCV
        before = len(df)
        df = df.dropna(subset=numeric_cols)
        dropped = before - len(df)
        if dropped:
            logger.info("Eliminadas %d filas con NaN", dropped)

        # 3. Validar coherencia OHLC  (high >= low, etc.)
        mask_invalid = (df["high"] < df["low"]) | (df["volume"] < 0)
        invalid_count = mask_invalid.sum()
        if invalid_count:
            logger.warning("Eliminadas %d filas con OHLC incoherente", invalid_count)
            df = df[~mask_invalid]

        # 4. Detectar outliers (marcar, no eliminar)
        df = DataCleaner.detect_outliers(df, column="close")

        # 5. Ordenar por timestamp
        if "open_time" in df.columns:
            df = df.sort_values("open_time").reset_index(drop=True)

        return df


# ──────────────────────────────────────────────────────────────────────
#  BinanceDataHandler — clase principal
# ──────────────────────────────────────────────────────────────────────

# Mapeo de intervalos Binance → python-binance constants
_INTERVAL_MAP: Dict[str, str] = {
    "1m": AsyncClient.KLINE_INTERVAL_1MINUTE,
    "3m": AsyncClient.KLINE_INTERVAL_3MINUTE,
    "5m": AsyncClient.KLINE_INTERVAL_5MINUTE,
    "15m": AsyncClient.KLINE_INTERVAL_15MINUTE,
    "30m": AsyncClient.KLINE_INTERVAL_30MINUTE,
    "1h": AsyncClient.KLINE_INTERVAL_1HOUR,
    "2h": AsyncClient.KLINE_INTERVAL_2HOUR,
    "4h": AsyncClient.KLINE_INTERVAL_4HOUR,
    "1d": AsyncClient.KLINE_INTERVAL_1DAY,
    "1w": AsyncClient.KLINE_INTERVAL_1WEEK,
}


class BinanceDataHandler:
    """
    Handler principal de datos de Binance Futures.

    Responsabilidades:
    - Conexión WebSocket para klines en tiempo real (multi-timeframe)
    - Descarga de datos históricos (hasta 3 años)
    - Limpieza y normalización
    - Persistencia en SQLite local

    Uso:
        handler = BinanceDataHandler()
        await handler.start()

        # Tiempo real
        df = handler.get_realtime_klines("BTCUSDT", "1m")

        # Históricos
        df = await handler.get_historical_data("BTCUSDT", "1h", "1 Jan 2023")

        await handler.stop()
    """

    DEFAULT_SYMBOL = "BTCUSDT"
    DEFAULT_TIMEFRAMES = ("1m", "5m", "1h", "4h")
    MAX_RECONNECT_ATTEMPTS = 10
    RECONNECT_BASE_DELAY = 2          # segundos (exponencial)

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[Tuple[str, ...]] = None,
        db_url: Optional[str] = None,
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        self.symbols = symbols or [self.DEFAULT_SYMBOL]
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES
        self.api_key = api_key or BINANCE_API_KEY
        self.api_secret = api_secret or BINANCE_API_SECRET

        # Almacenamiento
        self.store = DataStore(db_url=db_url)
        self.cleaner = DataCleaner()

        # AsyncClient & WebSocket
        self._client: Optional[AsyncClient] = None
        self._bsm: Optional[BinanceSocketManager] = None

        # Buffers en memoria: colas para appends en O(1)
        self._realtime_buffers: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=1500))  # type: ignore

        # Alt-Data Buffers (Microestructura)
        self._orderbook_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._funding_rates: Dict[str, float] = defaultdict(float)

        # Control
        self._ws_tasks: List[asyncio.Task] = []
        self._running = False
        self._reconnect_count = 0
        self._subscribers: List[Callable] = []

    # ══════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ══════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Inicializa la conexión async, la DB y lanza los WebSockets."""
        logger.info("Iniciando BinanceDataHandler...")

        # 1. SQLite
        self.store.initialize()

        # 2. AsyncClient — con DNS resiliente y retry
        self._client = await self._create_client_with_retry()

        # 3. BinanceSocketManager
        self._bsm = BinanceSocketManager(self._client)

        # 4. Lanzar un WebSocket por cada (symbol, timeframe)
        self._running = True
        for symbol in self.symbols:
            for tf in self.timeframes:
                task = asyncio.create_task(
                    self._ws_kline_loop(symbol, tf),
                    name=f"ws_{symbol}_{tf}",
                )
                self._ws_tasks.append(task)
                
        # 5. Alt-Data polling loop
        alt_data_task = asyncio.create_task(
            self._alt_data_loop(),
            name="alt_data_polling"
        )
        self._ws_tasks.append(alt_data_task)

        logger.info(
            "WebSocket activo para %d streams (%s × %s)",
            len(self._ws_tasks),
            self.symbols,
            list(self.timeframes),
        )

    async def _create_client_with_retry(self, max_retries: int = 5) -> AsyncClient:  # type: ignore
        """
        Crea el AsyncClient de Binance con resiliencia de grado institucional.

        Soluciona el bug conocido de aiodns/pycares en Windows donde c-ares
        no puede contactar los DNS del sistema operativo. Fuerza el uso del
        ThreadedResolver (nativo del OS) y aplica retry con backoff exponencial.
        """
        import aiohttp  # type: ignore
        import aiohttp.resolver  # type: ignore

        # ── Fix: Forzar ThreadedResolver como default para TODO el proceso ──
        # Cuando aiodns está instalado, aiohttp usa AsyncResolver (c-ares)
        # como DefaultResolver. En Windows, c-ares falla con
        # "Could not contact DNS servers" porque no lee la config DNS del OS.
        # Parcheamos el DefaultResolver a nivel de módulo para que
        # python-binance (que crea su propio ClientSession internamente)
        # use el resolver del sistema operativo automáticamente.
        aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver  # type: ignore
        logger.info("DNS Resolver parcheado → ThreadedResolver (OS-native)")

        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Conectando a Binance API (intento %d/%d)...",
                    attempt, max_retries,
                )
                client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                )
                logger.info("✅ Conexión a Binance API establecida exitosamente.")
                return client

            except Exception as exc:
                last_error = exc
                backoff = min(2 ** attempt, 30)  # 2, 4, 8, 16, 30 segundos
                logger.warning(
                    "⚠️ Intento %d/%d fallido: %s — reintentando en %ds...",
                    attempt, max_retries, exc, backoff,
                )
                await asyncio.sleep(backoff)

        # Si todos los intentos fallan, lanzar el error original
        raise ConnectionError(
            f"No se pudo conectar a Binance API después de {max_retries} intentos. "
            f"Último error: {last_error}"
        )

    async def stop(self) -> None:
        """Detiene los WebSockets y cierra el AsyncClient."""
        logger.info("Deteniendo BinanceDataHandler...")
        self._running = False

        for task in self._ws_tasks:
            task.cancel()
        await asyncio.gather(*self._ws_tasks, return_exceptions=True)
        self._ws_tasks.clear()

        if self._client:
            await self._client.close_connection()  # type: ignore
            self._client = None

        logger.info("BinanceDataHandler detenido.")

    # ══════════════════════════════════════════════════════════════════
    #  WebSocket — klines en tiempo real
    # ══════════════════════════════════════════════════════════════════

    async def _ws_kline_loop(self, symbol: str, interval: str) -> None:
        """
        Loop de reconexión automática para un stream de klines.
        Implementa back-off exponencial en caso de desconexión.
        """
        attempt = 0

        while self._running:
            try:
                attempt += 1
                logger.info(
                    "Conectando WebSocket kline %s %s (intento %d)...",
                    symbol,
                    interval,
                    attempt,
                )

                kline_socket = self._bsm.kline_futures_socket(  # type: ignore
                    symbol=symbol,
                    interval=_INTERVAL_MAP.get(interval, interval),
                )
                async with kline_socket as stream:
                    attempt = 0  # reset on successful connect
                    self._reconnect_count = 0
                    logger.info(
                        "✅ WebSocket conectado: %s %s", symbol, interval
                    )

                    while self._running:
                        msg = await asyncio.wait_for(stream.recv(), timeout=60)
                        if msg is None:
                            break

                        if "e" in msg and msg["e"] == "error":  # type: ignore
                            logger.error("Error en stream: %s", msg)
                            break

                        await self._handle_kline_msg(msg, symbol, interval)  # type: ignore

            except asyncio.CancelledError:
                logger.debug("WebSocket %s %s cancelado.", symbol, interval)
                return

            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout en WebSocket %s %s, reconectando...",
                    symbol,
                    interval,
                )

            except Exception as exc:
                logger.error(
                    "Error en WebSocket %s %s: %s — reconectando...",
                    symbol,
                    interval,
                    exc,
                )

            # ── Back-off exponencial ──
            if not self._running:
                return

            if attempt > self.MAX_RECONNECT_ATTEMPTS:
                logger.critical(
                    "❌ Máximo de reconexiones alcanzado para %s %s",
                    symbol,
                    interval,
                )
                return

            delay = min(self.RECONNECT_BASE_DELAY ** attempt, 300)
            logger.info("Reintentando en %.0f s...", delay)
            await asyncio.sleep(delay)

    async def _handle_kline_msg(
        self,
        msg: Dict[str, Any],
        symbol: str,
        interval: str,
    ) -> None:
        """Procesa un mensaje de kline, actualiza el buffer y persiste si cerrada."""
        kline_data = msg.get("k")
        if not kline_data:
            return

        cleaned = self.cleaner.clean_kline_row(kline_data)
        is_closed = cleaned.pop("is_closed", False)

        key = (symbol, interval)

        # Actualizar buffer en memoria (O(1) list operations)
        buf = self._realtime_buffers[key]
        
        # Si la vela tiene el mismo open_time que la anterior, actualizarla (vela abierta)
        if buf and buf[-1]["open_time"] == cleaned["open_time"]:
            buf[-1] = cleaned
        else:
            buf.append(cleaned)

        # Persistir solo velas cerradas
        if is_closed:
            new_row_df = pd.DataFrame([cleaned])
            self.store.save_klines(symbol, interval, new_row_df)

            # Notificar suscriptores
            for callback in self._subscribers:
                try:
                    callback(symbol, interval, cleaned)
                except Exception as exc:
                    logger.error("Error en subscriber: %s", exc)

    # ══════════════════════════════════════════════════════════════════
    #  Datos en tiempo real (Alt-Data & Order Book)
    # ══════════════════════════════════════════════════════════════════

    async def _alt_data_loop(self) -> None:
        """Loop asíncrono para pollear datos alternativos (OrderBook L1 y Funding) desde REST."""
        while self._running:
            try:
                for symbol in self.symbols:
                    # Traer orderbook tick (weight: 2)
                    ticker = await self._client.futures_book_ticker(symbol=symbol)  # type: ignore
                    bid_qty = float(ticker.get("bidQty", 0))
                    ask_qty = float(ticker.get("askQty", 0))
                    total_qty = bid_qty + ask_qty
                    imbalance = (bid_qty - ask_qty) / total_qty if total_qty > 0 else 0.0

                    self._orderbook_data[symbol] = {
                        "bidPrice": float(ticker.get("bidPrice", 0)),
                        "bidQty": bid_qty,
                        "askPrice": float(ticker.get("askPrice", 0)),
                        "askQty": ask_qty,
                        "imbalance": round(imbalance, 4)  # type: ignore
                    }

                    # Traer Funding Rate (via mark_price, weight: 1)
                    mark = await self._client.futures_mark_price(symbol=symbol)  # type: ignore
                    self._funding_rates[symbol] = float(mark.get("lastFundingRate", 0.0))

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug("Error polleando Alt-Data: %s", exc)

            await asyncio.sleep(2.0)  # Polling relajado cada 2s
            
    def get_alternative_data(self, symbol: str) -> Dict[str, Any]:
        """Retorna la microestructura actual para el símbolo (Imbalance y Funding Rate)."""
        return {
            "orderbook": self._orderbook_data.get(symbol, {}),
            "funding_rate": self._funding_rates.get(symbol, 0.0)
        }

    # ══════════════════════════════════════════════════════════════════
    #  Datos en tiempo real (Klines)
    # ══════════════════════════════════════════════════════════════════

    def get_realtime_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
    ) -> pd.DataFrame:
        """
        Retorna un DataFrame OHLCV con los datos en tiempo real
        acumulados en el buffer de memoria para el par/intervalo dado.

        Columns: open_time, open, high, low, close, volume,
                 close_time, quote_volume, num_trades
        """
        key = (symbol.upper(), interval)
        buf = self._realtime_buffers.get(key)

        if not buf:
            logger.debug("Sin datos en buffer para %s %s", symbol, interval)
            return pd.DataFrame(
                columns=[
                    "open_time", "open", "high", "low", "close",
                    "volume", "close_time", "quote_volume", "num_trades",
                ]
            )

        # Build DataFrame on the fly
        df = pd.DataFrame(list(buf))
        return self.cleaner.clean_data(df)

    # ══════════════════════════════════════════════════════════════════
    #  Datos históricos
    # ══════════════════════════════════════════════════════════════════

    async def get_historical_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Descarga datos históricos de Binance Futures.

        Args:
            symbol:     Par de trading (e.g. "BTCUSDT").
            interval:   Intervalo de velas ("1m", "5m", "1h", "4h", "1d").
            start_date: Fecha de inicio como string UTC ("1 Jan 2023")
                        o None para descargar los últimos 3 años.
            end_date:   Fecha de fin (None = ahora).

        Returns:
            DataFrame limpio con columnas OHLCV + timestamps.
        """
        if self._client is None:
            self._client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
            )

        if start_date is None:
            start_dt = datetime.now(timezone.utc) - timedelta(days=3 * 365)
            start_date = start_dt.strftime("%d %b %Y")

        bi_interval = _INTERVAL_MAP.get(interval, interval)

        logger.info(
            "Descargando históricos %s %s desde %s...",
            symbol,
            interval,
            start_date,
        )

        raw_klines = await self._client.get_historical_klines(  # type: ignore
            symbol=symbol,
            interval=bi_interval,
            start_str=start_date,
            end_str=end_date,
            klines_type=HistoricalKlinesType.FUTURES,
        )

        if not raw_klines:
            logger.warning("Sin datos históricos para %s %s", symbol, interval)
            return pd.DataFrame()

        # Cada kline cruda es una lista de 12 elementos:
        # [open_time, open, high, low, close, volume, close_time,
        #  quote_volume, num_trades, taker_buy_base, taker_buy_quote, ignore]
        df = pd.DataFrame(
            raw_klines,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )

        # Convertir timestamps (ms → datetime)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Convertir OHLCV a float
        for col in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[col] = df[col].astype(float)
        df["num_trades"] = df["num_trades"].astype(int)

        # Descartar columnas innecesarias
        df = df.drop(columns=["taker_buy_base", "taker_buy_quote", "ignore"])

        # Limpiar
        df = self.cleaner.clean_data(df)

        # Persistir
        saved = self.store.save_klines(symbol, interval, df)
        logger.info(
            "Descargados %d klines (%d nuevos guardados) [%s %s]",
            len(df),
            saved,
            symbol,
            interval,
        )

        return df

    # ══════════════════════════════════════════════════════════════════
    #  Limpieza (delegado público)
    # ══════════════════════════════════════════════════════════════════

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza completa del DataFrame.

        - Elimina NaN en OHLCV
        - Convierte a tipos numéricos
        - Detecta outliers por z-score
        - Valida coherencia (high >= low)
        - Ordena por open_time
        """
        return self.cleaner.clean_data(df)

    # ══════════════════════════════════════════════════════════════════
    #  Consultas locales (SQLite)
    # ══════════════════════════════════════════════════════════════════

    def get_dataframe(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Carga datos desde SQLite, limpia y retorna.
        Si hay un buffer en tiempo real, lo concatena al final.
        """
        df_stored = self.store.load_klines(symbol, interval, start, end)

        # Append real-time buffer desde memoria cruda (lista)
        key = (symbol, interval)
        buf = self._realtime_buffers.get(key)
        if buf:
            buf_df = pd.DataFrame(list(buf))
            df_stored = pd.concat([df_stored, buf_df], ignore_index=True)
            df_stored = df_stored.drop_duplicates(subset=["open_time"], keep="last")

        if df_stored.empty:
            return df_stored

        return self.cleaner.clean_data(df_stored)

    # ══════════════════════════════════════════════════════════════════
    #  Suscripciones
    # ══════════════════════════════════════════════════════════════════

    def subscribe(self, callback: Callable) -> None:
        """
        Registra un callback que se invoca con cada vela cerrada.
        Firma: callback(symbol: str, interval: str, kline: dict)
        """
        self._subscribers.append(callback)
        logger.debug("Suscriptor registrado. Total: %d", len(self._subscribers))

    def unsubscribe(self, callback: Callable) -> None:
        """Elimina un callback previamente registrado."""
        self._subscribers.remove(callback)

    # ══════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_streams(self) -> int:
        return len([t for t in self._ws_tasks if not t.done()])

    def buffer_sizes(self) -> Dict[str, int]:
        """Retorna el tamaño de cada buffer en memoria."""
        return {
            f"{sym}_{itv}": len(df)
            for (sym, itv), df in self._realtime_buffers.items()
        }

    def __repr__(self) -> str:
        status = "RUNNING" if self._running else "STOPPED"
        return (
            f"<BinanceDataHandler status={status} "
            f"symbols={self.symbols} timeframes={list(self.timeframes)} "
            f"streams={self.active_streams}>"
        )
