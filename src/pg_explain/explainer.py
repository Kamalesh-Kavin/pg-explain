"""
explainer.py — Runs EXPLAIN ANALYZE against Postgres and returns the raw plan.

Responsibilities:
  1. Hold a module-level asyncpg connection pool (created once at startup).
  2. Expose run_explain(sql, include_buffers) which:
       a. Acquires a connection from the pool.
       b. Opens a transaction with BEGIN.
       c. Sets a per-statement timeout so runaway queries can't hang the server.
       d. Runs EXPLAIN (ANALYZE, FORMAT JSON, [BUFFERS]) against the SQL.
       e. ALWAYs rolls back the transaction — even for SELECT statements.
          This is the critical safety rule: EXPLAIN ANALYZE *executes* the
          query for real, so a DELETE would delete rows if we committed.
       f. Returns the parsed JSON plan as a Python dict.

Why asyncpg instead of psycopg2 or SQLAlchemy?
  asyncpg is a pure-async Postgres driver with no synchronous fallback.
  It is the fastest Python Postgres driver available (~3x psycopg2) because:
    - It speaks the Postgres binary wire protocol directly (no string parsing).
    - It is fully non-blocking, so one server thread can handle many concurrent
      explain requests without blocking.
  The trade-off: it only works with asyncio. That is fine — FastAPI is async.

Why a connection pool?
  Creating a new TCP connection to Postgres takes ~1-3 ms (TLS handshake,
  authentication, session setup). With a pool we pay that cost once at startup
  and reuse connections for every request. asyncpg's built-in pool handles
  connection health checks and max-size limits automatically.

Connection pool sizing (MIN=1, MAX=5):
  pg-explain is a dev/learning tool, not a production service. 5 connections
  is more than enough. Keeping it small avoids holding Postgres connections
  open unnecessarily.
"""

import asyncpg
import json
import os
from typing import Any

# ---------------------------------------------------------------------------
# Module-level pool — created in init_pool(), closed in close_pool().
# FastAPI's lifespan handler calls both (see server.py).
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    """
    Create the asyncpg connection pool using DATABASE_URL from the environment.

    Called once when the FastAPI app starts (lifespan startup event).
    Raises if the DB is unreachable — better to fail fast at startup than
    to get confusing errors on the first request.
    """
    global _pool
    database_url = os.environ["DATABASE_URL"]  # hard fail if not set
    _pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,   # keep at least 1 connection alive at all times
        max_size=5,   # never open more than 5 simultaneous connections
    )


async def close_pool() -> None:
    """
    Gracefully close all connections in the pool.

    Called once when the FastAPI app shuts down (lifespan shutdown event).
    asyncpg.Pool.close() waits for in-flight queries to finish before
    closing, so we won't cut off a running EXPLAIN.
    """
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

async def run_explain(sql: str, include_buffers: bool = False) -> dict[str, Any]:
    """
    Execute EXPLAIN ANALYZE against `sql` and return the raw plan as a dict.

    Args:
        sql:             The SQL statement to analyse. Can be any statement
                         type — SELECT, INSERT, UPDATE, DELETE, CTEs, etc.
        include_buffers: If True, add BUFFERS TRUE to the EXPLAIN options.
                         Buffer stats show how many Postgres 8 KB pages were
                         served from shared_buffers (cache hit) vs read from
                         disk (cache miss). Useful for I/O diagnosis.

    Returns:
        The first element of the EXPLAIN JSON array — a dict with a
        "Plan" key containing the tree of plan nodes.

        Example shape:
            {
              "Plan": {
                "Node Type": "Seq Scan",
                "Relation Name": "users",
                "Startup Cost": 0.0,
                "Total Cost": 180.0,
                "Plan Rows": 10000,
                "Actual Rows": 9982,
                "Actual Total Time": 12.4,
                "Actual Loops": 1,
                ...
              },
              "Planning Time": 0.3,
              "Execution Time": 12.9
            }

    Raises:
        RuntimeError: if the pool has not been initialised yet.
        asyncpg.PostgresError: if Postgres rejects the SQL (syntax error, etc.)
        asyncpg.QueryCanceledError: if the statement exceeds the timeout.
    """
    if _pool is None:
        raise RuntimeError("Connection pool not initialised — call init_pool() first")

    # Build the EXPLAIN options string.
    # FORMAT JSON is mandatory: the text format is not machine-parseable.
    # ANALYZE TRUE means Postgres actually runs the query and adds real
    # timing / row counts to every node.
    buffers_option = ", BUFFERS TRUE" if include_buffers else ""
    explain_sql = f"EXPLAIN (ANALYZE TRUE, FORMAT JSON{buffers_option}) {sql}"

    async with _pool.acquire() as conn:
        # ------------------------------------------------------------------
        # SAFETY: wrap everything in a transaction that we ALWAYS roll back.
        #
        # Why this matters:
        #   EXPLAIN ANALYZE executes the query for real. If the user sends
        #   "DELETE FROM users WHERE id = 1", Postgres deletes that row
        #   during the explain run. Without a ROLLBACK the delete would be
        #   committed to disk.
        #
        #   By opening a transaction before the EXPLAIN and rolling it back
        #   unconditionally (the finally block), we guarantee:
        #     - Write statements execute (so EXPLAIN gets real timing data)
        #     - But their effects are never committed
        #
        # Note: this also means aggregate functions and sequences inside the
        # query may advance even though we roll back — that is a known
        # minor caveat, not fixable without wrapping at the sequence level.
        # ------------------------------------------------------------------
        await conn.execute("BEGIN")
        try:
            # Set a 5-second timeout for this statement only.
            # If a query takes longer than 5 s it is almost certainly
            # something a user should not be running via this tool.
            # The timeout is a session-level SET so it only applies to
            # statements issued on this connection until we ROLLBACK.
            await conn.execute("SET LOCAL statement_timeout = '5000ms'")

            # Run the actual EXPLAIN ANALYZE.
            # asyncpg returns each row as a Record object. EXPLAIN FORMAT JSON
            # always returns exactly one row with one column ("QUERY PLAN")
            # whose value is the JSON string.
            rows = await conn.fetch(explain_sql)

            # rows[0][0] is the raw JSON string from Postgres.
            # We parse it into a Python list — EXPLAIN FORMAT JSON always
            # returns a JSON array with one element (the top-level plan).
            plan_list: list[dict[str, Any]] = json.loads(rows[0][0])

            # Return just the first (and only) element of the array.
            return plan_list[0]

        finally:
            # ALWAYS roll back — whether the EXPLAIN succeeded or raised.
            # This is the unconditional safety net.
            await conn.execute("ROLLBACK")
