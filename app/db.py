"""Plain psycopg connection pool, used for the migrations table and any direct SQL."""

from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app import config

_username, _password = config.resolve_db_credentials()
_dsn = (
    f"host={config.DB_HOST} port={config.DB_PORT} dbname={config.DB_NAME} "
    f"user={_username} password={_password}"
)

pool = ConnectionPool(conninfo=_dsn, open=True)


@contextmanager
def get_cursor():
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur
