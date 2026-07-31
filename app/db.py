"""Plain psycopg connection pool, used for the migrations table and any direct SQL."""

from contextlib import contextmanager

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
    """Yields a dict-row psycopg cursor from the shared connection pool.

    Returns:
        A context manager yielding a psycopg cursor (rows as dicts); the underlying
        connection is returned to the pool on exit.
    """
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur
