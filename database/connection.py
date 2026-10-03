"""Database engine and session management.

The engine is created from ``DATABASE_URL``; switching from SQLite to
PostgreSQL is a connection-string change (``postgresql+psycopg://...``) plus
installing the driver. SQLite-specific pragmas are only applied to SQLite.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - driver callback
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


def create_db_engine(url: str) -> Engine:
    """Create an engine; ensures the SQLite directory exists."""
    parsed = make_url(url)
    kwargs: dict = {"future": True, "pool_pre_ping": True}
    if parsed.get_backend_name() == "sqlite":
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if parsed.get_backend_name() == "sqlite":
        event.listen(engine, "connect", _sqlite_pragmas)
    return engine


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    return create_db_engine(url or get_settings().database_url)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def database_exists(url: str) -> bool:
    """True if the database file (SQLite) exists or the server accepts a connection."""
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite":
        db = parsed.database
        return bool(db) and (db == ":memory:" or Path(db).exists())
    try:
        with create_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - any connection failure means "not available"
        logger.exception("Database connection check failed")
        return False


def check_connection(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Database connection failed")
        return False


def existing_tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())
