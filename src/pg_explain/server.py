"""
server.py — FastAPI application: defines routes and wires everything together.

Architecture:
  POST /explain  →  validate request (Pydantic)
                 →  run_explain()  (explainer.py — hits Postgres)
                 →  analyze()      (analyzer.py — pure Python, no DB)
                 →  build ExplainResponse and return as JSON

  GET  /health   →  check DB connectivity and return {"status": "ok"}

Lifespan (startup/shutdown):
  FastAPI's lifespan context manager is the recommended way to run code once
  at startup and once at shutdown. We use it to:
    - init_pool() on startup   — creates the asyncpg connection pool
    - close_pool() on shutdown — gracefully drains and closes all connections

  The older @app.on_event("startup") / @app.on_event("shutdown") decorators
  still work but are deprecated as of FastAPI 0.95.

Error handling:
  - 400 Bad Request  — if Postgres rejects the SQL (syntax error, unknown table…)
  - 408 Request Timeout — if the statement exceeds the 5-second timeout
  - 500 Internal Server Error — unexpected failures (pool not ready, etc.)

  We catch asyncpg exceptions explicitly because they carry useful Postgres
  error messages that we want to pass back to the caller rather than letting
  FastAPI turn them into generic 500 errors.
"""

import asyncpg
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pg_explain.explainer import init_pool, close_pool, run_explain
from pg_explain.analyzer import analyze
from pg_explain.models import ExplainRequest, ExplainResponse


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and once at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.

    Why use lifespan instead of global startup/shutdown events?
      lifespan is the modern FastAPI approach (v0.95+). It uses a single
      async context manager which is cleaner, testable, and aligns with
      how ASGI servers (uvicorn) manage app lifecycle.
    """
    # --- STARTUP ---
    await init_pool()   # opens the asyncpg connection pool
    yield
    # --- SHUTDOWN ---
    await close_pool()  # drains and closes all DB connections


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="pg-explain",
    description="Query Plan Visualiser — runs EXPLAIN ANALYZE and surfaces insights",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health_check():
    """
    Returns {"status": "ok", "db": "connected"} if Postgres is reachable.

    This is intentionally lightweight — it only imports run_explain's pool,
    not run_explain itself. We do a minimal 'SELECT 1' to confirm the DB
    connection is alive without running a full EXPLAIN cycle.

    Use this endpoint for liveness probes in Docker / Kubernetes, or just
    to confirm the server started correctly before sending real queries.
    """
    from pg_explain.explainer import _pool
    if _pool is None:
        raise HTTPException(status_code=503, detail="DB pool not initialised")
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ok", "db": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unreachable: {exc}")


@app.post("/explain", response_model=ExplainResponse, summary="Analyse a SQL query")
async def explain(request: ExplainRequest) -> ExplainResponse:
    """
    Run EXPLAIN ANALYZE on the provided SQL and return a structured analysis.

    The heavy lifting is split across two modules:
      - explainer.py  — handles the DB interaction (runs the query, ROLLBACK)
      - analyzer.py   — pure Python tree walk, no DB calls needed

    This separation makes analyzer.py fully unit-testable without a real DB.

    Request body:
        sql            (str, required) — the SQL to analyse
        include_buffers (bool, default False) — add BUFFERS to EXPLAIN options

    Response:
        See ExplainResponse in models.py for the full shape.

    Error responses:
        400 — Postgres syntax/semantic error in the provided SQL
        408 — Query exceeded the 5-second statement timeout
        500 — Unexpected internal error
    """
    try:
        # Step 1: Hit Postgres. run_explain handles BEGIN/ROLLBACK safety.
        raw_plan = await run_explain(request.sql, request.include_buffers)
    except asyncpg.QueryCanceledError:
        # Statement exceeded SET LOCAL statement_timeout = '5000ms'.
        # 408 Request Timeout is the most appropriate HTTP status here.
        raise HTTPException(
            status_code=408,
            detail="Query exceeded the 5-second statement timeout.",
        )
    except asyncpg.PostgresError as exc:
        # Postgres rejected the SQL — syntax error, unknown table, bad type, etc.
        # exc.args[0] contains the Postgres error message (human-readable).
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # Pool not initialised — should not happen in normal operation.
        raise HTTPException(status_code=500, detail=str(exc))

    # Step 2: Walk the plan tree and extract warnings/stats.
    # This is pure Python — no DB calls, no I/O.
    result = analyze(raw_plan)

    # Step 3: Build the one-liner summary for the top of the response.
    # Format: "<Root Node Type> | est. cost <X> | actual <Y> ms"
    summary = (
        f"{result.top_node_type} | "
        f"est. cost {result.total_cost:.2f} | "
        f"actual {result.total_actual_ms:.2f} ms"
    )

    # Step 4: Assemble and return the Pydantic response model.
    # FastAPI will serialise this to JSON automatically.
    return ExplainResponse(
        summary=summary,
        warnings=result.warnings,
        total_cost=result.total_cost,
        total_actual_ms=result.total_actual_ms,
        node_count=result.node_count,
        slowest_node=result.slowest_node,
        misestimated_nodes=result.misestimated_nodes,
        raw_plan=raw_plan,
    )
