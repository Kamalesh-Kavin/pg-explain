# pg-explain — Query Plan Visualiser

**Project 1 of 27** from an 18-month systems engineering learning roadmap.

Runs `EXPLAIN ANALYZE` against a Postgres query and returns a structured JSON
summary of the plan — warnings, cost estimates, timing, and row misestimation
detection — without requiring any client-side knowledge of the EXPLAIN output format.

---

## What it does

Send a SQL query to `POST /explain`. Get back:

- A one-line `summary` (top node type, estimated cost, actual runtime)
- Plain-English `warnings` for common problems: Seq Scans on large tables, stale
  planner statistics, slow nodes, sorts that spilled to disk
- `total_cost` (planner estimate, unitless), `total_actual_ms` (real wall-clock time)
- `node_count` (depth of the plan tree)
- `slowest_node` (the single node that consumed the most real time)
- `misestimated_nodes` (nodes where actual rows / estimated rows > 10x)
- The full `raw_plan` JSON for your own analysis

---

## Architecture

```
Client
  |
  | POST /explain  {"sql": "...", "include_buffers": false}
  v
FastAPI (server.py)
  |
  |-- ExplainRequest validated by Pydantic
  |
  v
explainer.py
  |  asyncpg connection pool
  |  BEGIN
  |  SET LOCAL statement_timeout = '5000ms'
  |  EXPLAIN (ANALYZE TRUE, FORMAT JSON) <sql>
  |  ROLLBACK  ← always, even for DELETE/UPDATE
  v
analyzer.py  (pure Python, no DB calls)
  |  recursive tree walk
  |  apply warning rules to every node
  v
ExplainResponse  (Pydantic model → JSON)
```

### Key design decisions

| Decision | Reason |
|---|---|
| `asyncpg` instead of `psycopg2` | Binary wire protocol, fully async, ~3x faster |
| `BEGIN` + `ROLLBACK` wrapping | `EXPLAIN ANALYZE` executes writes for real; ROLLBACK prevents data loss |
| `SET LOCAL statement_timeout` | Prevents runaway queries from blocking the server |
| `FORMAT JSON` | Only machine-parseable EXPLAIN format; text format is not |
| `analyzer.py` separate from `explainer.py` | Keeps DB logic and analysis logic independently testable |
| `actual_time × loops` for node timing | `Actual Total Time` is per-loop; must multiply by `Actual Loops` for real cost |

---

## Warning rules

| Check | Threshold | Message |
|---|---|---|
| Seq Scan | actual rows > 1,000 | Suggest adding an index |
| Row misestimation | actual / estimated > 10x (either direction) | Run `ANALYZE` to refresh stats |
| Slow node | actual ms > 100 ms | Investigate this sub-tree |
| Sort spill to disk | `Sort Method` contains "external" | Increase `work_mem` |

---

## Stack

- **Python 3.13** with `uv` as the package manager
- **FastAPI** — async HTTP framework
- **asyncpg** — async Postgres driver (binary protocol)
- **Pydantic v2** — request/response validation and serialisation
- **python-dotenv** — loads `DATABASE_URL` from `.env`
- **pytest** + **pytest-asyncio** — unit testing

---

## Running locally

### Prerequisites

- Docker (for the Postgres container)
- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 1. Start Postgres

```bash
docker run -d \
  --name pg-explain-db \
  -e POSTGRES_USER=pgexplain \
  -e POSTGRES_PASSWORD=pgexplain \
  -e POSTGRES_DB=pgexplain \
  -p 5435:5432 \
  postgres:16-alpine
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — the default values match the container above
```

### 3. Install dependencies and run

```bash
uv pip install -e .
uv run uvicorn pg_explain.server:app --reload --port 8001
```

### 4. Try it

```bash
# Health check
curl http://localhost:8001/health

# Analyse a query
curl -s http://localhost:8001/explain \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT * FROM users WHERE age > 30"}' | python3 -m json.tool
```

---

## Running tests

```bash
uv run pytest tests/ -v
```

All 20 tests run without a database. The analyzer is tested with synthetic plan
dicts that mirror real Postgres `EXPLAIN FORMAT JSON` output.

---

## API reference

### `GET /health`

Returns `{"status": "ok", "db": "connected"}` if Postgres is reachable.

### `POST /explain`

**Request body:**

```json
{
  "sql": "SELECT * FROM users WHERE age > 30",
  "include_buffers": false
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `sql` | string | yes | — | SQL query to analyse |
| `include_buffers` | bool | no | `false` | Add `BUFFERS TRUE` to EXPLAIN options |

**Response:**

```json
{
  "summary": "Seq Scan | est. cost 209.00 | actual 8.28 ms",
  "warnings": [
    "Seq Scan on \"users\" returned 8,457 rows — consider adding an index..."
  ],
  "total_cost": 209.0,
  "total_actual_ms": 8.276,
  "node_count": 1,
  "slowest_node": { "node_type": "Seq Scan", "actual_ms": 7.83 },
  "misestimated_nodes": [],
  "raw_plan": { ... }
}
```

**Error responses:**

| Status | Cause |
|---|---|
| 400 | Postgres rejected the SQL (syntax error, unknown table, etc.) |
| 408 | Query exceeded the 5-second statement timeout |
| 503 | DB pool not initialised or DB unreachable |

---

## What I learned building this

- `EXPLAIN` is a prediction. `EXPLAIN ANALYZE` actually executes the query.
  Always wrap in a transaction and `ROLLBACK` — especially for `DELETE`/`UPDATE`.
- Cost numbers are unitless planner units (based on `seq_page_cost=1.0` and
  `random_page_cost=4.0`), not milliseconds. Never compare costs to time.
- `Actual Total Time` is per-loop, not total. For nodes inside Nested Loops
  the real time is `Actual Total Time × Actual Loops`.
- Row misestimation > 10x usually means stale statistics → `ANALYZE` the table.
  The planner picks join strategies (Hash Join vs Nested Loop) based on row
  estimates, so bad estimates can cause catastrophically wrong plan choices.
- Index Scan ≠ Index Only Scan. Index Only Scan avoids the heap entirely
  (all columns in the index), but degrades to Index Scan on heavily-updated
  tables (visibility map not all-visible).
- Bitmap Heap Scan is a hybrid: collect matching TIDs from the index first,
  then fetch heap pages in physical order. Better than Index Scan for medium
  selectivity.
- Sort `"external merge"` means the sort spilled to a temp file because
  `work_mem` was too small. Increasing `work_mem` for the session fixes it.
- `asyncpg` speaks the Postgres binary wire protocol directly — no string
  parsing, fully non-blocking. The pool reuses connections across requests
  (creating a new TCP connection costs ~1-3 ms each time).

---

## Project structure

```
pg-explain/
├── src/pg_explain/
│   ├── __init__.py     # empty — makes it a Python package
│   ├── server.py       # FastAPI app: POST /explain, GET /health
│   ├── explainer.py    # asyncpg pool + EXPLAIN ANALYZE + ROLLBACK safety
│   ├── analyzer.py     # recursive plan tree walker + warning rules
│   └── models.py       # Pydantic request/response models
├── tests/
│   └── test_analyzer.py  # 20 unit tests — no DB required
├── .env.example
├── .gitignore
└── pyproject.toml
```
