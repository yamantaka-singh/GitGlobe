"""Measure /search against the committed eval set.

Phase 4's exit criterion is recall@10 > 0.7, and without a number here every
tuning decision is taste. Run this against whatever /search currently does
*before* changing it, so a fix has a baseline to beat — a change that cannot be
shown to beat substring matching has not been demonstrated to work.

    python api/tests/eval_search.py --api http://127.0.0.1:8001

`--ceiling` reports how much of the eval set the corpus can answer at all. An
expected repository that was never ingested is unreachable by any retrieval
method, so recall has to be read against that ceiling rather than against 1.0.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

EVAL_SET = pathlib.Path(__file__).with_name("eval_set.json")

#: Name lookups, kept separate from `eval_set.json` on purpose.
#:
#: Every one of the 30 eval queries describes what software *does*; not one
#: looks a repository up by name. That blind spot hid a real regression:
#: replacing the `ILIKE` fallback with full-text search made repository names
#: unsearchable, because Postgres tokenises `ohmyzsh/ohmyzsh` as one `file`
#: token rather than the word `ohmyzsh`, and recall@10 did not move at all.
#:
#: These are checked at rank 1, not rank 10. Searching a repository by its exact
#: name has one right answer and it belongs at the top; finding it ninth is a
#: failure that recall@10 would happily score as a pass.
NAME_LOOKUPS = [
    ("ohmyzsh", "ohmyzsh/ohmyzsh"),
    ("react", "react/react"),
    ("vue", "vuejs/vue"),
    ("linux", "torvalds/linux"),
    ("kubernetes", "kubernetes/kubernetes"),
    ("tensorflow", "tensorflow/tensorflow"),
]


def search(api: str, query: str, limit: int) -> list[str]:
    url = f"{api}/search?q={urllib.parse.quote(query)}&limit={limit}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    return [hit["repo"]["full_name"] for hit in payload[:limit]]


def recall_at_k(got: list[str], expected: list[str]) -> float:
    """Fraction of the expected repositories that appear in the top k.

    Recall rather than precision because the eval set lists several acceptable
    answers per query and does not claim to list every one — precision would
    punish a correct result simply for being unlisted.
    """
    if not expected:
        return 0.0
    hits = {name.lower() for name in got} & {name.lower() for name in expected}
    return len(hits) / len(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--ceiling", action="store_true",
                        help="also report what fraction of expected repos exist in Postgres")
    args = parser.parse_args()

    cases = json.loads(EVAL_SET.read_text())
    reachable = corpus_names(cases) if args.ceiling else None

    total = 0.0
    empty = 0
    print(f"{'recall':>7}  {'hits':>5}  query")
    print("-" * 60)
    for case in cases:
        got = search(args.api, case["query"], args.k)
        score = recall_at_k(got, case["expected"])
        total += score
        empty += not got
        found = int(round(score * len(case["expected"])))
        print(f"{score:>7.2f}  {found:>2}/{len(case['expected']):<2}  {case['query']}")

    n = len(cases)
    print("-" * 60)
    print(f"recall@{args.k} = {total / n:.3f} over {n} queries   (target > 0.7)")
    print(f"queries returning nothing at all: {empty}/{n}")
    if reachable is not None:
        print(f"ceiling (expected repos present in corpus): {reachable:.3f}")

    print(f"\n{'at 1':>7}  name lookup")
    print("-" * 60)
    first = 0
    for query, expected in NAME_LOOKUPS:
        got = search(args.api, query, args.k)
        ok = bool(got) and got[0].lower() == expected.lower()
        first += ok
        print(f"{'yes' if ok else 'NO':>7}  {query} -> {got[0] if got else '(nothing)'}")
    print("-" * 60)
    print(f"exact name at rank 1: {first}/{len(NAME_LOOKUPS)}")


def corpus_names(cases: list[dict]) -> float:
    """What fraction of expected repositories exist in `repo` at all."""
    import subprocess

    wanted = sorted({name for case in cases for name in case["expected"]})
    values = ",".join("(" + sql_quote(n) + ")" for n in wanted)
    sql = (
        f"SELECT count(*) FROM (VALUES {values}) AS w(name) "
        f"WHERE EXISTS (SELECT 1 FROM repo r WHERE lower(r.full_name) = lower(w.name))"
    )
    out = subprocess.run(
        ["docker", "exec", "gitglobe-postgres", "psql", "-U", "gitglobe", "-d", "gitglobe", "-tAc", sql],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip() or 0) / len(wanted)


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    main()
