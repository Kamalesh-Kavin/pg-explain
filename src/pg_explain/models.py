"""
models.py — Pydantic request/response shapes for pg-explain.

Why Pydantic?
  - FastAPI uses Pydantic models to automatically validate incoming JSON
    and serialize outgoing responses. If a field is the wrong type, FastAPI
    returns a 422 before our code even runs.
  - We're using Pydantic v2 (pydantic >= 2.0) which is ~5-50x faster than v1
    thanks to the Rust-based pydantic-core under the hood.

Design decisions:
  - ExplainRequest is intentionally minimal: just the SQL string and a flag
    for buffer stats. Keeping the surface area small makes the API easy to
    understand and extend later.
  - ExplainResponse is a *summary* layer on top of the raw plan JSON.
    The raw plan is still included (raw_plan) so callers can do their own
    analysis if they want. But the fields above it save them the work of
    walking the tree themselves.
  - NodeSummary is a tiny helper model for the "slowest node" field. It
    only carries the fields we actually surface in the response.
  - MisestimatedNode captures the three numbers needed to understand a
    planner estimate gone wrong: what it predicted, what actually happened,
    and by what ratio.
"""

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# REQUEST
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    """
    Body sent by the caller to POST /explain.

    Fields:
        sql           — the query to analyse. Can be any SQL that Postgres
                        accepts: SELECT, INSERT, UPDATE, DELETE, CTEs, etc.
                        Even write statements are safe because explainer.py
                        wraps everything in BEGIN … ROLLBACK.

        include_buffers — when True we add BUFFERS TRUE to the EXPLAIN
                        options. This tells Postgres to report how many
                        8 KB disk pages were read from cache vs disk for
                        each node. Useful for diagnosing I/O problems but
                        adds noise for simple queries, so it defaults False.
    """
    sql: str = Field(..., min_length=1, description="SQL query to analyse")
    include_buffers: bool = Field(
        default=False,
        description="Include BUFFERS option in EXPLAIN ANALYZE",
    )


# ---------------------------------------------------------------------------
# RESPONSE HELPERS
# ---------------------------------------------------------------------------

class NodeSummary(BaseModel):
    """
    A minimal description of a single plan node.

    We only need two fields for the 'slowest_node' use-case:
      node_type  — e.g. "Hash Join", "Seq Scan"
      actual_ms  — wall-clock time this node took (actual_time × loops)
    """
    node_type: str
    actual_ms: float


class MisestimatedNode(BaseModel):
    """
    A plan node where the planner's row estimate was badly wrong.

    Why does this matter?
      The query planner picks join strategies (Hash Join vs Nested Loop)
      and memory allocations based on row estimates. If estimates are off
      by 10x or more the planner might choose a strategy that is orders
      of magnitude slower than optimal. This usually means table statistics
      are stale — run ANALYZE on the table to fix it.

    Fields:
        node_type   — e.g. "Seq Scan"
        table_name  — relation name (None for nodes that don't scan a table)
        estimated   — rows the planner predicted
        actual      — rows that actually came out
        ratio       — actual / estimated, rounded to 1 decimal place
    """
    node_type: str
    table_name: str | None
    estimated: float
    actual: float
    ratio: float


# ---------------------------------------------------------------------------
# RESPONSE
# ---------------------------------------------------------------------------

class ExplainResponse(BaseModel):
    """
    The full response returned by POST /explain.

    Fields:
        summary            — a one-line human-readable description of the
                             top-level plan node and total cost.

        warnings           — a list of plain-English problem descriptions
                             detected by analyzer.py. Empty list = no issues.

        total_cost         — the planner's estimated cost for the whole query
                             (unitless planner units — NOT milliseconds).

        total_actual_ms    — real wall-clock execution time in milliseconds,
                             as measured by Postgres. This IS reliable.

        node_count         — how many plan nodes the tree contains. Deeply
                             nested trees (high count) can be hard to optimise.

        slowest_node       — the single node that consumed the most real time.
                             None if EXPLAIN ANALYZE didn't return timing data.

        misestimated_nodes — all nodes where actual rows / estimated rows > 10.
                             Empty list = planner statistics are healthy.

        raw_plan           — the raw EXPLAIN ANALYZE FORMAT JSON output from
                             Postgres. Callers can do their own analysis with
                             this if our summary is not enough.
    """
    summary: str
    warnings: list[str]
    total_cost: float
    total_actual_ms: float
    node_count: int
    slowest_node: NodeSummary | None
    misestimated_nodes: list[MisestimatedNode]
    raw_plan: dict[str, Any]
