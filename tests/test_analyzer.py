"""
test_analyzer.py — Unit tests for analyzer.py.

These tests exercise the warning rules and stat extraction WITHOUT a real
database. We construct synthetic plan dicts that look exactly like what
Postgres returns from EXPLAIN FORMAT JSON, then assert on the AnalysisResult.

Why test analyzer separately from explainer?
  The whole point of splitting explainer.py (DB) from analyzer.py (logic)
  is testability. We can run these tests with zero infrastructure — no
  Docker, no Postgres. They run instantly and deterministically.

What these tests cover:
  1. Basic stats extraction (top node type, cost, timing, node count)
  2. Seq Scan warning fires for large tables
  3. Seq Scan warning does NOT fire for small tables (no false positives)
  4. Row misestimation warning fires when actual/estimated > 10x
  5. Row misestimation warning fires when estimated/actual > 10x (reversed)
  6. Slow node warning fires for sub-tree nodes > 100 ms
  7. Sort spill-to-disk warning fires when Sort Method contains "external"
  8. Multi-node tree: node_count is correct and slowest_node is right
  9. No warnings on a clean plan (all thresholds respected)
"""

import pytest
from pg_explain.analyzer import analyze


# ---------------------------------------------------------------------------
# Helper — factory for a minimal plan node dict
# ---------------------------------------------------------------------------

def make_node(
    node_type: str = "Seq Scan",
    relation_name: str | None = None,
    total_cost: float = 100.0,
    plan_rows: int = 100,
    actual_rows: int = 100,
    actual_total_time: float = 10.0,
    actual_loops: int = 1,
    sort_method: str | None = None,
    parent_relationship: str | None = "Outer",  # None = root node
    plans: list | None = None,
) -> dict:
    """
    Build a synthetic plan node that mirrors Postgres EXPLAIN FORMAT JSON output.
    Only include keys we care about — Postgres includes many more but analyzer.py
    uses .get() with defaults so missing keys are safe.
    """
    node: dict = {
        "Node Type": node_type,
        "Total Cost": total_cost,
        "Plan Rows": plan_rows,
        "Actual Rows": actual_rows,
        "Actual Total Time": actual_total_time,
        "Actual Loops": actual_loops,
    }
    if relation_name is not None:
        node["Relation Name"] = relation_name
    if sort_method is not None:
        node["Sort Method"] = sort_method
    if parent_relationship is not None:
        # Root node does NOT have "Parent Relationship" — only child nodes do.
        node["Parent Relationship"] = parent_relationship
    if plans is not None:
        node["Plans"] = plans
    return node


def make_plan(root_node: dict, execution_time: float = 10.0) -> dict:
    """
    Wrap a root node in the top-level plan dict that analyze() expects.
    The root node must NOT have "Parent Relationship" (pass parent_relationship=None).
    """
    return {
        "Plan": root_node,
        "Planning Time": 0.1,
        "Execution Time": execution_time,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicStats:
    """analyzer.py correctly extracts top-level statistics."""

    def test_top_node_type(self):
        root = make_node(node_type="Index Scan", parent_relationship=None)
        result = analyze(make_plan(root))
        assert result.top_node_type == "Index Scan"

    def test_total_cost(self):
        root = make_node(total_cost=450.25, parent_relationship=None)
        result = analyze(make_plan(root))
        assert result.total_cost == 450.25

    def test_total_actual_ms_from_execution_time(self):
        # "Execution Time" from the plan should be used, not the root node time.
        root = make_node(actual_total_time=5.0, parent_relationship=None)
        result = analyze(make_plan(root, execution_time=12.9))
        assert result.total_actual_ms == 12.9

    def test_single_node_count(self):
        root = make_node(parent_relationship=None)
        result = analyze(make_plan(root))
        assert result.node_count == 1

    def test_multi_node_count(self):
        # Root with two children = 3 nodes total.
        child1 = make_node(node_type="Seq Scan", actual_total_time=1.0)
        child2 = make_node(node_type="Hash", actual_total_time=1.0)
        root = make_node(
            node_type="Hash Join",
            parent_relationship=None,
            plans=[child1, child2],
        )
        result = analyze(make_plan(root))
        assert result.node_count == 3


class TestSeqScanWarning:
    """Seq Scan warning fires only when actual rows > 1000."""

    def test_large_seq_scan_triggers_warning(self):
        root = make_node(
            node_type="Seq Scan",
            relation_name="users",
            actual_rows=9_982,
            parent_relationship=None,
        )
        result = analyze(make_plan(root))
        assert len(result.warnings) >= 1
        assert any("Seq Scan" in w and "users" in w for w in result.warnings)

    def test_small_seq_scan_no_warning(self):
        # 500 rows is below the 1000-row threshold — should be no seq scan warning.
        root = make_node(
            node_type="Seq Scan",
            relation_name="small_table",
            actual_rows=500,
            parent_relationship=None,
        )
        result = analyze(make_plan(root))
        seq_scan_warnings = [w for w in result.warnings if "Seq Scan" in w]
        assert len(seq_scan_warnings) == 0

    def test_exactly_at_threshold_no_warning(self):
        # Exactly 1000 rows — NOT over the threshold, so no warning.
        root = make_node(
            node_type="Seq Scan",
            relation_name="borderline",
            actual_rows=1_000,
            parent_relationship=None,
        )
        result = analyze(make_plan(root))
        seq_scan_warnings = [w for w in result.warnings if "Seq Scan" in w]
        assert len(seq_scan_warnings) == 0


class TestMisestimationWarning:
    """Row misestimation warning fires when ratio > 10x in either direction."""

    def test_underestimate_fires_warning(self):
        # Planner predicted 100, got 9982 → ratio = 99.8x (> 10x threshold)
        root = make_node(
            node_type="Seq Scan",
            relation_name="users",
            plan_rows=100,
            actual_rows=9_982,
            parent_relationship=None,
        )
        result = analyze(make_plan(root))
        assert len(result.misestimated_nodes) == 1
        assert result.misestimated_nodes[0].estimated == 100
        assert result.misestimated_nodes[0].actual == 9_982

    def test_overestimate_fires_warning(self):
        # Planner predicted 10000, got 5 → ratio = 2000x (> 10x threshold)
        root = make_node(
            node_type="Index Scan",
            plan_rows=10_000,
            actual_rows=5,
            parent_relationship=None,
        )
        result = analyze(make_plan(root))
        assert len(result.misestimated_nodes) == 1
        # ratio should be displayed as actual/estimated when actual > estimated,
        # or estimated/actual when estimated > actual (always >= 1)
        assert result.misestimated_nodes[0].ratio >= 10

    def test_accurate_estimate_no_warning(self):
        # Planner predicted 1000, got 1050 → ratio = 1.05x (< 10x)
        root = make_node(
            node_type="Seq Scan",
            plan_rows=1_000,
            actual_rows=1_050,
            parent_relationship=None,
        )
        result = analyze(make_plan(root))
        assert len(result.misestimated_nodes) == 0

    def test_zero_plan_rows_no_crash(self):
        # plan_rows=0 could cause division by zero — should be handled gracefully.
        root = make_node(
            node_type="Seq Scan",
            plan_rows=0,
            actual_rows=500,
            parent_relationship=None,
        )
        result = analyze(make_plan(root))  # must not raise
        assert len(result.misestimated_nodes) == 0


class TestSlowNodeWarning:
    """Slow node warning fires for child nodes > 100 ms. Root is excluded."""

    def test_slow_child_fires_warning(self):
        # 150 ms child inside a Hash Join — should warn.
        slow_child = make_node(
            node_type="Seq Scan",
            relation_name="orders",
            actual_total_time=150.0,  # per loop
            actual_loops=1,
            actual_rows=100,
            plan_rows=100,
        )
        root = make_node(
            node_type="Hash Join",
            parent_relationship=None,
            actual_total_time=200.0,
            plans=[slow_child],
        )
        result = analyze(make_plan(root, execution_time=200.0))
        slow_warnings = [w for w in result.warnings if "Slow node" in w]
        assert len(slow_warnings) >= 1

    def test_fast_child_no_warning(self):
        fast_child = make_node(
            node_type="Index Scan",
            actual_total_time=5.0,
        )
        root = make_node(
            node_type="Hash Join",
            parent_relationship=None,
            plans=[fast_child],
        )
        result = analyze(make_plan(root))
        slow_warnings = [w for w in result.warnings if "Slow node" in w]
        assert len(slow_warnings) == 0

    def test_slow_root_does_not_fire_warning(self):
        # Root node is slow but we don't fire "Slow node" for it — the
        # total_actual_ms field already surfaces this to the caller.
        root = make_node(
            node_type="Seq Scan",
            parent_relationship=None,  # no parent = root
            actual_total_time=500.0,
            actual_rows=100,
            plan_rows=100,
        )
        result = analyze(make_plan(root, execution_time=500.0))
        slow_warnings = [w for w in result.warnings if "Slow node" in w]
        assert len(slow_warnings) == 0


class TestSortSpillWarning:
    """Sort spill-to-disk warning fires when Sort Method contains 'external'."""

    def test_external_sort_fires_warning(self):
        sort_node = make_node(
            node_type="Sort",
            sort_method="external merge",
        )
        root = make_node(
            node_type="Gather Merge",
            parent_relationship=None,
            plans=[sort_node],
        )
        result = analyze(make_plan(root))
        spill_warnings = [w for w in result.warnings if "spilled to disk" in w]
        assert len(spill_warnings) == 1

    def test_in_memory_sort_no_warning(self):
        sort_node = make_node(
            node_type="Sort",
            sort_method="quicksort",
        )
        root = make_node(
            node_type="Gather Merge",
            parent_relationship=None,
            plans=[sort_node],
        )
        result = analyze(make_plan(root))
        spill_warnings = [w for w in result.warnings if "spilled to disk" in w]
        assert len(spill_warnings) == 0


class TestSlowestNode:
    """slowest_node is the node with the highest actual_total_time × actual_loops."""

    def test_slowest_node_identified_correctly(self):
        fast = make_node(node_type="Index Scan", actual_total_time=5.0, actual_loops=1)
        slow = make_node(node_type="Seq Scan", actual_total_time=80.0, actual_loops=1)
        root = make_node(
            node_type="Hash Join",
            parent_relationship=None,
            actual_total_time=90.0,
            plans=[fast, slow],
        )
        result = analyze(make_plan(root))
        # Root has 90 ms, but Seq Scan child has 80 ms.
        # Root is the slowest (90 > 80).
        assert result.slowest_node is not None
        assert result.slowest_node.node_type == "Hash Join"
        assert result.slowest_node.actual_ms == 90.0

    def test_loops_multiplier_applied(self):
        # A node with 10 ms × 20 loops = 200 ms effective time.
        # That should beat the root (50 ms × 1).
        loopy_child = make_node(
            node_type="Index Scan",
            actual_total_time=10.0,
            actual_loops=20,
        )
        root = make_node(
            node_type="Nested Loop",
            parent_relationship=None,
            actual_total_time=50.0,
            actual_loops=1,
            plans=[loopy_child],
        )
        result = analyze(make_plan(root))
        assert result.slowest_node is not None
        assert result.slowest_node.node_type == "Index Scan"
        assert result.slowest_node.actual_ms == 200.0


class TestCleanPlan:
    """A well-optimised plan should produce no warnings."""

    def test_no_warnings_on_clean_index_scan(self):
        root = make_node(
            node_type="Index Scan",
            relation_name="users",
            plan_rows=10,
            actual_rows=10,
            actual_total_time=0.05,
            actual_loops=1,
            parent_relationship=None,
        )
        result = analyze(make_plan(root, execution_time=0.1))
        assert result.warnings == []
        assert result.misestimated_nodes == []
