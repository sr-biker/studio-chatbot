"""Minimal migration runner: applies db_migrations/*.sql in filename order, tracked in
schema_migrations. The FAQ vector store itself is managed by langchain-postgres (PGVector
creates its own collection/embedding tables on first use) — nothing to migrate there."""

import logging

from app import config
from app.db import pool

log = logging.getLogger(__name__)


def run_migrations() -> None:
    with pool.connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (filename text PRIMARY KEY, applied_at timestamptz DEFAULT now())"
        )
        applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()}

        for path in sorted(config.MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            sql = path.read_text().replace("{vector_dimension}", str(config.VECTOR_DIMENSION))
            log.info("Applying migration %s", path.name)
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
        conn.commit()
