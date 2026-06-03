"""
Database bootstrap for Echo2Equation.

- Reads DB settings from environment variables (set via deployments/.env).
- Creates a SQLAlchemy engine and session factory.
- Exposes `Base` for your table classes (in tables.py).
- Provides `get_db()` FastAPI dependency for per-request sessions.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ---- Environment (Docker compose sets these from deployments/.env)
DB_USER = os.getenv("DB_USER", "echo2eq")
DB_PASS = os.getenv("DB_PASS", "echo2eq_pw")
DB_HOST = os.getenv("DB_HOST", "db")        # service name in docker-compose
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "echo2eq_db")

# Example: postgresql+psycopg2://user:pass@db:5432/echo2eq_db
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ---- SQLAlchemy core objects
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # automatically test connections; avoids stale sockets
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# Base class for your ORM tables (see backend/database/tables.py)
Base = declarative_base()


def get_db():
    """
    FastAPI dependency: yields a database session and ensures it closes.
    Usage:
        def route(dep: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
