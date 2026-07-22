"""Plain psycopg connection pool, used for the migrations table and any direct SQL."""

from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings

_username, _password = settings.resolve_db_credentials()
_dsn = (
    f"host={settings.db_host} port={settings.db_port} dbname={settings.db_name} "
    f"user={_username} password={_password}"
)

pool = ConnectionPool(conninfo=_dsn, open=True)


@contextmanager
def get_cursor():
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur
