"""
analyzer.py — Walks a Postgres EXPLAIN ANALYZE plan tree and extracts insights.

Responsibilities:
  1. Walk the nested plan node tree recursively.
  2. Collect statistics: total cost, total actual time, node count, slowest node.
  3. Apply warning rules to every node and accumulate human-readable warnings.
  4. Return a structured AnalysisResult that server.py converts to ExplainResponse.

How EXPLAIN FORMAT JSON structures the tree:
  The top-level dict looks like:
    {
      "Plan": { <node> },
      "Planning Time": 0.3,
      "Execution Time": 12.9
    }

  Each node can have a "Plans" key with a list of child nodes:
    {
      "Node Type": "Hash Join",
      "Startup Cost": 0.0,
      "Total Cost": 450.25,
      "Plan Rows": 1000,        ← planner estimate
      "Actual Rows": 987,       ← real row count (from ANALYZE)
      "Actual Total Time": 28.4,← cumulative time including children
      "Actual Loops": 2,        ← node may execute multiple times (e.g. nested loop)
      "Plans": [ { <child node> }, ... ]
    }

Key timing subtlety — "Actual Total Time" × "Actual Loops":
  "Actual Total Time" is the *per-loop* time. If a node runs 3 times
  (Actual Loops = 3) the real total for that node is:
    Actual Total Time × Actual Loops
  This matters for nodes inside Nested Loop joins. We use this formula
  everywhere so the numbers are comparable across different node types.

Warning rules (decided in the brainstorm):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Rule                │ Threshold         │ Message                       │
  ├─────────────────────┼───────────────────┼───────────────────────────────┤
  │ Seq Scan            │ actual rows > 1000│ suggest adding an index       │
  │ Row misestimation   │ actual/est > 10x  │ stale stats, run ANALYZE      │
  │ Slow node           │ actual_ms > 100ms │ node took a long time         │
  │ Sort spill to disk  │ "external" in     │ increase work_mem             │
  │                     │ Sort Method       │                               │
  └─────────────────────┴───────────────────┴───────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Any

from pg_explain.models import MisestimatedNode, NodeSummary


# ---------------------------------------------------------------------------
# Internal result container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    All extracted data from a single EXPLAIN plan.

    This is an internal dataclass — it is not exposed directly via the API.
    server.py maps it into an ExplainResponse Pydantic model.

    Using a dataclass (not Pydantic) here because:
      - This is purely internal computation — no JSON serialisation needed.
      - dataclass is lighter and clearer for in-memory data containers.
    """
    top_node_type: str          # type of the root plan node, e.g. "Hash Join"
    total_cost: float           # planner's estimated cost for the whole query
    total_actual_ms: float      # real execution time (Execution Time field)
    node_count: int             # total number of nodes in the tree
    warnings: list[str] = field(default_factory=list)
    slowest_node: NodeSummary | None = None
    misestimated_nodes: list[MisestimatedNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Thresholds — centralised so they are easy to tweak
# ---------------------------------------------------------------------------

SEQ_SCAN_ROW_THRESHOLD = 1_000   # warn if a seq scan returns more than this
MISESTIMATE_RATIO_THRESHOLD = 10  # warn if actual/estimated rows > this
SLOW_NODE_MS_THRESHOLD = 100.0    # warn if a node takes longer than this


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(plan: dict[str, Any]) -> AnalysisResult:
    """
    Analyse a parsed EXPLAIN ANALYZE JSON plan and return an AnalysisResult.

    Args:
        plan: The dict returned by explainer.run_explain() — i.e. the first
              element of the EXPLAIN FORMAT JSON array.

              Expected top-level keys:
                "Plan"           — the root plan node (required)
                "Execution Time" — total wall-clock ms (optional but always
                                   present with ANALYZE TRUE)

    Returns:
        An AnalysisResult with all extracted stats and warnings.
    """
    root_node = plan["Plan"]

    # "Execution Time" is the total wall-clock time Postgres reports for the
    # entire query. It includes overhead outside the plan tree (e.g. result
    # serialisation) so it is the most accurate "how long did this query take"
    # number. We fall back to the root node's timing if somehow absent.
    execution_time_ms: float = plan.get("Execution Time", 0.0)

    # Mutable accumulators passed into the recursive walk.
    # Using a list as a "box" so the inner function can mutate them without
    # needing nonlocal declarations everywhere.
    all_nodes: list[dict[str, Any]] = []
    warnings: list[str] = []
    misestimated: list[MisestimatedNode] = []
    slowest_ms: list[float] = [0.0]
    slowest_node: list[NodeSummary | None] = [None]

    # Walk the entire tree, collecting data.
    _walk(root_node, all_nodes, warnings, misestimated, slowest_ms, slowest_node)

    return AnalysisResult(
        top_node_type=root_node.get("Node Type", "Unknown"),
        total_cost=root_node.get("Total Cost", 0.0),
        total_actual_ms=execution_time_ms,
        node_count=len(all_nodes),
        warnings=warnings,
        slowest_node=slowest_node[0],
        misestimated_nodes=misestimated,
    )


# ---------------------------------------------------------------------------
# Recursive tree walker
# ---------------------------------------------------------------------------

def _walk(
    node: dict[str, Any],
    all_nodes: list[dict[str, Any]],
    warnings: list[str],
    misestimated: list[MisestimatedNode],
    slowest_ms: list[float],
    slowest_node: list[NodeSummary | None],
) -> None:
    """
    Recursively visit every node in the plan tree.

    This function uses a pre-order traversal (process parent before children)
    which is the natural order for plan trees — the root is the last operation
    executed (output), children are the input operations. Pre-order makes the
    warnings list roughly match the visual order in EXPLAIN output.

    Args:
        node:         The current plan node dict.
        all_nodes:    Accumulator — every node visited is appended here.
        warnings:     Accumulator — warning strings are appended here.
        misestimated: Accumulator — MisestimatedNode objects appended here.
        slowest_ms:   Single-element list boxing the current max node time.
        slowest_node: Single-element list boxing the current slowest NodeSummary.
    """
    all_nodes.append(node)

    node_type: str = node.get("Node Type", "Unknown")

    # ------------------------------------------------------------------
    # Extract timing for this node.
    #
    # "Actual Total Time" is per-loop time in milliseconds.
    # "Actual Loops" is how many times this node executed.
    # Real node time = Actual Total Time × Actual Loops.
    #
    # Exception: for the root node, "Actual Total Time" already covers
    # everything — but the formula still works (loops=1 at root).
    # ------------------------------------------------------------------
    actual_time_per_loop: float = node.get("Actual Total Time", 0.0)
    actual_loops: int = node.get("Actual Loops", 1)
    node_actual_ms: float = actual_time_per_loop * actual_loops

    # Track the slowest node seen so far.
    if node_actual_ms > slowest_ms[0]:
        slowest_ms[0] = node_actual_ms
        slowest_node[0] = NodeSummary(
            node_type=node_type,
            actual_ms=round(node_actual_ms, 2),
        )

    # ------------------------------------------------------------------
    # Row counts for misestimation check.
    #
    # "Plan Rows" = planner estimate (before running).
    # "Actual Rows" = real row count (from ANALYZE), per loop.
    # Total actual rows = Actual Rows × Actual Loops.
    # ------------------------------------------------------------------
    plan_rows: float = node.get("Plan Rows", 0.0)
    actual_rows_per_loop: float = node.get("Actual Rows", 0.0)
    actual_rows: float = actual_rows_per_loop * actual_loops

    # ------------------------------------------------------------------
    # WARNING RULE 1: Seq Scan on a large table.
    #
    # A Seq Scan reads every row in the table. For small tables this is
    # fine (often faster than an index scan). For large tables it is a
    # red flag — there might be a missing index.
    #
    # We only warn when actual rows > SEQ_SCAN_ROW_THRESHOLD (1000) to
    # avoid false positives on small lookup tables.
    # ------------------------------------------------------------------
    if node_type == "Seq Scan" and actual_rows > SEQ_SCAN_ROW_THRESHOLD:
        table_name = node.get("Relation Name", "unknown")
        warnings.append(
            f"Seq Scan on \"{table_name}\" returned {int(actual_rows):,} rows "
            f"— consider adding an index if this table is frequently queried."
        )

    # ------------------------------------------------------------------
    # WARNING RULE 2: Row count misestimation.
    #
    # If the planner predicted N rows but got M rows and M/N > 10,
    # the statistics are probably stale. Run ANALYZE on the table to
    # update them. Stale stats lead to bad join strategy choices.
    #
    # Guard: skip if plan_rows is 0 to avoid division by zero.
    # Also skip if actual_rows is 0 — no data to misestimate against.
    # ------------------------------------------------------------------
    if plan_rows > 0 and actual_rows > 0:
        ratio = actual_rows / plan_rows
        if ratio > MISESTIMATE_RATIO_THRESHOLD or ratio < (1 / MISESTIMATE_RATIO_THRESHOLD):
            # Use the bigger of the two as the "ratio" so it is always >= 1
            display_ratio = ratio if ratio > 1 else 1 / ratio
            table_name = node.get("Relation Name")  # None for non-scan nodes
            misestimated.append(MisestimatedNode(
                node_type=node_type,
                table_name=table_name,
                estimated=plan_rows,
                actual=actual_rows,
                ratio=round(display_ratio, 1),
            ))
            warnings.append(
                f"Row misestimation on {node_type}"
                + (f" \"{table_name}\"" if table_name else "")
                + f": planner predicted {int(plan_rows):,} rows, "
                f"got {int(actual_rows):,} ({display_ratio:.1f}x off) "
                f"— run ANALYZE to refresh statistics."
            )

    # ------------------------------------------------------------------
    # WARNING RULE 3: Slow node (> 100 ms).
    #
    # Any single node that takes > 100 ms deserves attention. The root
    # node's time includes all children so we skip it to avoid a
    # duplicate "the whole query was slow" warning — callers can see
    # total_actual_ms for that. We detect "root" by checking if the node
    # has no "Parent Relationship" key (only child nodes have that).
    #
    # Note: "Actual Total Time" for non-root nodes is cumulative
    # (includes sub-tree time), so a slow child might partly be due to
    # its own children. This is expected behaviour for EXPLAIN output.
    # ------------------------------------------------------------------
    is_root = "Parent Relationship" not in node
    if not is_root and node_actual_ms > SLOW_NODE_MS_THRESHOLD:
        warnings.append(
            f"Slow node: {node_type} took {node_actual_ms:.1f} ms "
            f"— investigate this sub-tree."
        )

    # ------------------------------------------------------------------
    # WARNING RULE 4: Sort spilled to disk.
    #
    # When Postgres sorts more data than work_mem allows, it spills to
    # a temporary disk file ("external merge"). This is much slower than
    # an in-memory sort and shows up in "Sort Method" containing the
    # word "external".
    #
    # Fix: increase work_mem for the session or globally, or add an
    # index that provides pre-sorted rows.
    # ------------------------------------------------------------------
    if node_type == "Sort":
        sort_method: str = node.get("Sort Method", "")
        if "external" in sort_method.lower():
            warnings.append(
                f"Sort spilled to disk (method: \"{sort_method}\") "
                f"— increase work_mem or add a supporting index."
            )

    # ------------------------------------------------------------------
    # Recurse into children.
    #
    # "Plans" is the key for child nodes in EXPLAIN FORMAT JSON.
    # It is absent on leaf nodes (Seq Scan, Index Scan, etc.).
    # ------------------------------------------------------------------
    for child in node.get("Plans", []):
        _walk(child, all_nodes, warnings, misestimated, slowest_ms, slowest_node)
