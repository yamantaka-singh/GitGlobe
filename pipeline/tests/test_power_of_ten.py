"""NASA's Power of 10, enforced — not aspired to.

Gerard Holzmann's rules were written for C in flight software. Three of the ten
are meaningless in Python: no heap allocation after init, sparing preprocessor
use, and single-level pointer dereferencing. The other seven translate directly,
and this file checks them on every test run.

A coding standard nobody checks is a document, not a standard. The value here is
not the rules — it is that violating one fails the build, and that every
exception is written down next to a reason someone can argue with.

**Where the rules are applied loosely, and why.** Rule 5 says two runtime
assertions per function. Applied literally that means 106 new assertions in this
codebase, most of them restating a type annotation. That is noise, and noise
trains people to stop reading assertions. The rule's intent — check what can
actually be wrong — is enforced instead at *boundaries*: anything parsing
external data, spending money, or writing an artifact another program reads.

Rule 3 (no dynamic allocation) has no Python analogue, but its *purpose* —
bounded, predictable memory — does. `test_no_unbounded_accumulation` is the
nearest honest equivalent.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SRC = Path(__file__).resolve().parents[1] / "src" / "gitglobe"

MAX_FUNCTION_LINES = 60

#: Rule 4 exceptions. Each needs a reason that survives being read aloud.
#: "It is long" is not a reason. Splitting a function purely to satisfy a line
#: count, where the parts have no independent meaning, makes code worse.
LONG_FUNCTION_ALLOWLIST = {
    ("cli.py", "main"):
        "An argparse table. Every line declares one flag; splitting it scatters "
        "the CLI's shape across helpers and makes it harder to see, not easier.",
    ("cli.py", "_cmd_doctor"):
        "Sequential report formatting. The parts have no independent meaning "
        "and no caller would ever want one of them alone.",
    ("tiles/build.py", "build_world"):
        "TRACKED, NOT ACCEPTED. Genuinely does three jobs — tiles, graph, "
        "manifest — and should be split. Deferred because the edge pipeline is "
        "mid-debug and this is the one function whose output the TypeScript "
        "verifier checks end to end. Refactoring it while its inputs are still "
        "changing risks a silent format regression to satisfy a line count.",
    ("phase2.py", "stage_edges"):
        "TRACKED, NOT ACCEPTED. Same reason: actively being debugged against a "
        "live BigQuery schema. Split once dependency edges land.",
    ("brain/features.py", "build_features"):
        "One flat dict of column definitions. Each entry is a single "
        "expression, and grouping them into helpers would hide the feature "
        "list, which is the thing a reader has come to see.",
    ("ingest/github.py", "_post"):
        "One HTTP call with its full error taxonomy inline — 502 cost, 403 "
        "rate limit, token rotation, backoff. The handling only makes sense "
        "next to the call it guards.",
    ("project/cluster.py", "cluster"):
        "A single linear pipeline: HDBSCAN, centroids, domains, noise "
        "assignment. Each step consumes the previous one's output.",
    ("embed/vertex.py", "embed_batch"):
        "A single batch processing flow that interacts with Vertex AI Batch Prediction API: "
        "writes JSONL, uploads to GCS, triggers job, polls, and parses results. Splitting it "
        "would fragment the single cohesive state machine into disparate helper functions.",
    ("phase2.py", "stage_embed"):
        "A single logical pipeline step that orchestrates either local synchronous execution "
        "or remote batch execution depending on scale. It reads repos, dispatches, and collects "
        "results. Splitting the 63 lines obscures the branching condition.",
    ("project/spherical.py", "project"):
        "Orchestrates UMAP manifold projection. The addition of sub-sampling for large graphs "
        "(50k+) naturally sits here before UMAP initialization to prevent OOM errors. It's "
        "better to keep the 62-line configuration contiguous.",
}
# `flow.py::ingest_repositories` and `phase2.py::stage_build` were here until
# the length check started excluding docstrings. Both are comfortably under the
# limit in code and needed no exception at all — the allowlist was carrying two
# entries that existed only because the measurement was wrong.

#: Rule 1 exceptions. Recursion is allowed only with a proven depth bound.
RECURSION_ALLOWLIST = {
    ("cli.py", "_plain"):
        "Tree-walks a JSON-shaped report. Bounded by an explicit `depth > 8` "
        "ceiling that returns a placeholder, so it cannot reach Python's "
        "recursion limit. An iterative rewrite would be less readable.",
}

#: Modules where external data enters the process. Rule 7 (validate every
#: parameter, check every return) is enforced strictly here and nowhere else,
#: because this is where wrong data actually comes from.
BOUNDARY_MODULES = {
    "tiles/format.py", "tiles/build.py", "brain/rubric.py",
    "brain/features.py", "embed/vertex.py", "graph/pagerank.py",
}


def source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def relative(path: Path) -> str:
    return str(path.relative_to(SRC))


def functions_in(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def code_lines(fn) -> int:
    """Function length EXCLUDING its docstring.

    Rule 4 exists so a reader can hold a function's control flow in their head
    at once. A docstring is not control flow. Counting it means a function that
    explains a subtle decision is penalised against one that does not, which is
    the opposite of what the rule is for — and it pushes people toward silence.

    This sharpens the rule rather than relaxing it. The test of that claim is
    whether the allowlist shrank: two entries became unnecessary when this
    changed, and every genuinely long function stayed on it.
    """
    total = (fn.end_lineno or fn.lineno) - fn.lineno + 1
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        doc = fn.body[0]
        total -= (doc.end_lineno or doc.lineno) - doc.lineno + 1
    return total


class TestRule1NoUnboundedRecursion(unittest.TestCase):
    def test_recursion_is_absent_or_depth_bounded(self) -> None:
        offenders = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for fn in functions_in(tree):
                calls_self = any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == fn.name
                    for n in ast.walk(fn)
                )
                if calls_self and (relative(path), fn.name) not in RECURSION_ALLOWLIST:
                    offenders.append(f"{relative(path)}::{fn.name}")
        self.assertEqual(
            offenders, [],
            "Unbounded recursion. Rewrite iteratively, or add a depth ceiling "
            "and an entry in RECURSION_ALLOWLIST explaining the bound.",
        )

    def test_allowlisted_recursion_still_has_a_ceiling(self) -> None:
        # An allowlist entry is a claim. Check the claim.
        for (filename, name), reason in RECURSION_ALLOWLIST.items():
            tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
            fn = next(f for f in functions_in(tree) if f.name == name)
            has_ceiling = any(
                isinstance(n, ast.Compare) and any(
                    isinstance(c, ast.Constant) and isinstance(c.value, int)
                    for c in n.comparators
                )
                for n in ast.walk(fn)
            )
            with self.subTest(f"{filename}::{name}"):
                self.assertTrue(has_ceiling, f"no depth comparison found; reason claims: {reason}")


class TestRule2BoundedLoops(unittest.TestCase):
    """Every loop must terminate for a reason a reader can point at."""

    @staticmethod
    def _is_bounded(node: ast.While) -> bool:
        # Bounded if the body can leave: an explicit break, a return, a raise,
        # or a guard that raises. `for` loops over finite iterables are bounded
        # by construction and are not checked.
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Break, ast.Return, ast.Raise)):
                return True
        return False

    def test_every_while_loop_can_terminate(self) -> None:
        offenders = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.While) and not self._is_bounded(node):
                    offenders.append(f"{relative(path)}:{node.lineno}")
        self.assertEqual(
            offenders, [],
            "`while` with no break, return, or raise. Add an explicit iteration "
            "ceiling that raises — a loop whose bound is 'the data is well "
            "formed' is not bounded.",
        )

    def test_retry_loops_are_range_bounded(self) -> None:
        # Retry against a paid API is the highest-consequence loop in the
        # codebase: unbounded, it bills forever. These must be `for _ in range`.
        for filename in ("embed/vertex.py", "brain/teacher.py", "ingest/github.py"):
            tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.While):
                    continue
                with self.subTest(f"{filename}:{node.lineno}"):
                    self.assertTrue(
                        self._is_bounded(node),
                        "network loop without a provable exit",
                    )


class TestRule4FunctionLength(unittest.TestCase):
    def test_functions_fit_on_a_page(self) -> None:
        offenders = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for fn in functions_in(tree):
                length = code_lines(fn)
                key = (relative(path), fn.name)
                if length > MAX_FUNCTION_LINES and key not in LONG_FUNCTION_ALLOWLIST:
                    offenders.append(f"{relative(path)}::{fn.name} ({length} code lines)")
        self.assertEqual(
            offenders, [],
            f"Over {MAX_FUNCTION_LINES} lines. Split it, or add it to "
            "LONG_FUNCTION_ALLOWLIST with a reason that is not 'it is long'.",
        )

    def test_the_allowlist_has_no_stale_entries(self) -> None:
        # An allowlist that outlives its entries stops meaning anything.
        stale = []
        for (filename, name) in LONG_FUNCTION_ALLOWLIST:
            path = SRC / filename
            if not path.exists():
                stale.append(f"{filename} (gone)")
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            match = [f for f in functions_in(tree) if f.name == name]
            if not match:
                stale.append(f"{filename}::{name} (gone)")
            elif code_lines(match[0]) <= MAX_FUNCTION_LINES:
                stale.append(f"{filename}::{name} (now short enough — remove it)")
        self.assertEqual(stale, [], "Stale allowlist entries.")

    def test_tracked_exceptions_are_labelled_honestly(self) -> None:
        # Two entries say the function SHOULD be split and is only deferred.
        # Keeping that distinction visible stops "deferred" becoming "fine".
        tracked = [k for k, v in LONG_FUNCTION_ALLOWLIST.items() if "TRACKED" in v]
        self.assertGreaterEqual(len(tracked), 2)
        for key in tracked:
            self.assertIn("Deferred" if key[1] == "build_world" else "Split",
                          LONG_FUNCTION_ALLOWLIST[key])


#: Functions taking arrays that must be the same length, and the arguments that
#: must agree. This is rule 7 made specific.
#:
#: A blanket "two assertions per function" would have added 106 assertions here,
#: most restating a type annotation, and noise teaches people to skip reading
#: assertions. The rule's *intent* is to check what can actually be wrong. In
#: array code the answer is precise: a length mismatch between parallel arrays
#: does not raise. numpy broadcasts, `zip` truncates, boolean masks shorten.
#: The result is a silently wrong globe rather than a stack trace.
PARALLEL_ARRAY_CONTRACTS = {
    ("tiles/format.py", "build_undirected_csr"): ("src", "dst", "weight"),
    ("tiles/build.py", "cluster_manifest_entries"): ("cluster_id", "theta", "phi", "domain"),
    ("graph/pagerank.py", "importance_order"): ("rank", "tiebreak"),
    ("graph/pagerank.py", "pagerank"): ("src", "dst"),
    ("brain/features.py", "reduce_embeddings"): ("vectors",),
}


class TestRule7ValidateAtBoundaries(unittest.TestCase):
    """Rule 7, aimed at the one failure mode that does not announce itself."""

    def test_parallel_array_functions_check_their_lengths(self) -> None:
        offenders = []
        for (filename, name), args in PARALLEL_ARRAY_CONTRACTS.items():
            tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
            match = [f for f in functions_in(tree) if f.name == name]
            if not match:
                offenders.append(f"{filename}::{name} (function is gone)")
                continue
            fn = match[0]
            # A length guard is a `len(...)` inside a comparison that can raise.
            guards_length = any(
                isinstance(n, ast.Raise)
                and any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name)
                    and c.func.id in ("len", "ValueError")
                    for c in ast.walk(fn)
                )
                for n in ast.walk(fn)
            )
            has_len_compare = any(
                isinstance(n, ast.Compare)
                and any(
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "len"
                    for sub in ast.walk(n)
                )
                for n in ast.walk(fn)
            )
            if not (guards_length and has_len_compare):
                offenders.append(f"{filename}::{name} (declares {args})")
        self.assertEqual(
            offenders, [],
            "Parallel-array function with no length check. numpy broadcasts, "
            "zip truncates, boolean masks shorten — none of them raise. The "
            "symptom is a wrong globe, not an exception.",
        )

    def test_the_guards_actually_fire(self) -> None:
        """A guard nobody exercises is a comment. Run each one."""
        import numpy as np

        from gitglobe.brain.features import reduce_embeddings
        from gitglobe.graph.pagerank import combine_edges, importance_order, pagerank
        from gitglobe.tiles.build import cluster_manifest_entries
        from gitglobe.tiles.format import build_undirected_csr

        with self.subTest("build_undirected_csr ragged"), self.assertRaises(ValueError):
            build_undirected_csr(3, [0, 1], [1], [1.0])
        with self.subTest("build_undirected_csr endpoint out of range"):
            with self.assertRaises(ValueError):
                build_undirected_csr(2, [0], [99], [1.0])
        with self.subTest("importance_order short tiebreak"), self.assertRaises(ValueError):
            importance_order(np.zeros(5), np.zeros(3))
        with self.subTest("pagerank ragged"), self.assertRaises(ValueError):
            pagerank(3, np.array([0, 1]), np.array([1]))
        with self.subTest("combine_edges ragged layer"), self.assertRaises(ValueError):
            combine_edges([(np.array([0, 1]), np.array([1]), np.array([1.0]), 1.0)])
        with self.subTest("cluster_manifest_entries ragged"), self.assertRaises(ValueError):
            cluster_manifest_entries(
                np.zeros(3, np.int32), np.zeros(2), np.zeros(3), np.zeros(3, np.uint8)
            )
        with self.subTest("reduce_embeddings wrong basis dimension"):
            _, basis = reduce_embeddings(np.random.default_rng(0).normal(size=(20, 16)), 4)
            with self.assertRaises(ValueError):
                reduce_embeddings(np.zeros((5, 32)), 4, basis=basis)


class TestRule7ConcurrencyIsolation(unittest.TestCase):
    """Rule 7 in async code: one task's failure must not destroy the others.

    No AST rule found this. `asyncio.gather` propagates the FIRST exception and
    cancels every remaining task, so a single malformed API response — or a
    single `assert_no_popularity` firing on a README that quotes `"stars": 42` —
    discards the entire unpersisted batch of a paid run.
    """

    PAID_PATHS = ("embed/vertex.py", "brain/teacher.py")

    def test_gather_never_runs_without_return_exceptions(self) -> None:
        offenders = []
        for filename in self.PAID_PATHS:
            tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "gather"
                ):
                    continue
                guarded = any(
                    kw.arg == "return_exceptions"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if not guarded:
                    offenders.append(f"{filename}:{node.lineno}")
        self.assertEqual(
            offenders, [],
            "asyncio.gather without return_exceptions=True on a path that "
            "spends money. One task's exception cancels the rest and loses "
            "everything since the last checkpoint.",
        )

    def test_workers_contain_their_own_failures(self) -> None:
        # The real defence: the worker catches, so the batch keeps its results.
        # `return_exceptions=True` is the backstop, not the plan.
        for filename in self.PAID_PATHS:
            tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
            workers = [
                fn for fn in functions_in(tree)
                if fn.name == "worker" and isinstance(fn, ast.AsyncFunctionDef)
            ]
            with self.subTest(filename):
                self.assertTrue(workers, "expected an async worker coroutine")
                for fn in workers:
                    self.assertTrue(
                        any(isinstance(n, ast.Try) for n in ast.walk(fn)),
                        "worker does not contain its own failures",
                    )

    def test_external_json_is_parsed_defensively(self) -> None:
        # A 200 does not guarantee the body's shape. Chained subscripts on a
        # response are how a KeyError escapes into `gather`.
        source = (SRC / "embed/vertex.py").read_text(encoding="utf-8")
        self.assertIn("except (KeyError, IndexError, TypeError, ValueError)", source)


class TestRule6MinimalScope(unittest.TestCase):
    def test_no_module_level_mutable_state(self) -> None:
        offenders = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                for target in targets:
                    if not isinstance(target, ast.Name) or target.id.isupper():
                        continue  # SCREAMING_CASE is a declared constant
                    if isinstance(value, (ast.Dict, ast.List, ast.Set)):
                        offenders.append(f"{relative(path)}:{node.lineno} {target.id}")
        self.assertEqual(
            offenders, [],
            "Module-level mutable state. Two runs in one process would share "
            "it, which is how a test suite passes and a real run does not.",
        )


class TestRule3Analogue(unittest.TestCase):
    """No dynamic allocation has no Python analogue; bounded memory does."""

    def test_streaming_paths_do_not_accumulate_without_limit(self) -> None:
        # At 1M repos an unbounded accumulator is gigabytes. The paid paths
        # checkpoint instead, so a crash costs one batch, not the run.
        for filename in ("embed/vertex.py", "brain/teacher.py"):
            source = (SRC / filename).read_text(encoding="utf-8")
            with self.subTest(filename):
                self.assertIn("CHECKPOINT_EVERY", source)
                self.assertIn("on_batch", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
