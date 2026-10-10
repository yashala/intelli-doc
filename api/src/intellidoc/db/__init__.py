from __future__ import annotations

from importlib.resources import files

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(files("intellidoc.db").joinpath("schema.sql").read_text())
    conn.commit()


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def create_pool(url: str, max_size: int = 5) -> ConnectionPool:
    with psycopg.connect(url) as conn:
        apply_schema(conn)
    return ConnectionPool(url, min_size=1, max_size=max_size, configure=_configure, open=True)
