from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest
from pgvector.psycopg import register_vector

from intellidoc.db import apply_schema
from intellidoc.embeddings import FastEmbedder


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


@pytest.fixture
def conn(database_url: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(database_url) as connection:
        apply_schema(connection)
        register_vector(connection)
        connection.execute("TRUNCATE documents CASCADE")
        connection.commit()
        yield connection


@pytest.fixture(scope="session")
def embedder() -> FastEmbedder:
    return FastEmbedder()
