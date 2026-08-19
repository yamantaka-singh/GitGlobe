"""The two silent failures in hybrid search, guarded.

No DB, no Qdrant, no credentials — both of these are pure functions over shapes.
Run directly: `python api/tests/test_search_contract.py`.

Both bugs below shipped, and neither raised anything a user or a log would show.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gitglobe_api.embed import (
    DIMENSIONS,
    INPUT_TYPE,
    EmbeddingUnavailable,
    vector_from,
)
from gitglobe_api.main import reciprocal_rank_fusion, to_or_tsquery


def nim_response(values):
    return {"data": [{"embedding": values, "index": 0}]}


def test_query_vector_matches_corpus_contract():
    """2048-d and unit L2 norm — the exact bug that made search impossible.

    The corpus is 2048-d. An earlier query path produced 1024-d `voyage-2`
    vectors, Qdrant rejected them on dimension, the exception was swallowed, and
    every search degraded to a substring match. Nothing anywhere said so, which
    is why the width is asserted at the boundary rather than trusted.
    """
    raw = [3.0] * DIMENSIONS  # deliberately un-normalised, as the model returns
    vector = vector_from(nim_response(raw))

    assert len(vector) == DIMENSIONS, f"expected {DIMENSIONS}-d, got {len(vector)}"
    norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))
    assert abs(norm - 1.0) < 1e-5, f"query vector must be L2-normalised, norm={norm}"

    # A wrong width must raise rather than reach Qdrant and be swallowed there.
    for wrong in (768, 1024, DIMENSIONS - 1):
        try:
            vector_from(nim_response([1.0] * wrong))
        except EmbeddingUnavailable as exc:
            assert str(DIMENSIONS) in str(exc)
        else:
            raise AssertionError(f"{wrong}-d vector was accepted")


def test_query_uses_the_query_subspace():
    """`query`, not `passage` — the failure that leaves no trace at all.

    The model is asymmetric: the same text as `passage` and as `query` has
    cosine 0.67, so they are different vectors. The corpus was written as
    `passage`. Sending `passage` here would return a correctly-shaped 2048-d
    vector, pass every width check above, and simply make search quietly worse
    forever. There is no runtime signal for it, so it is pinned here.
    """
    assert INPUT_TYPE == "query", (
        f"INPUT_TYPE is {INPUT_TYPE!r}; the corpus is 'passage', so queries "
        "must be 'query' or retrieval silently degrades"
    )


def test_zero_vector_does_not_become_nan():
    """A zero vector must survive normalisation. NaN poisons the Qdrant query."""
    vector = vector_from(nim_response([0.0] * DIMENSIONS))
    assert not np.isnan(vector).any(), "zero vector normalised to NaN"


class Hit:
    """Stand-in for a Qdrant ScoredPoint, which RRF must accept alongside dicts."""

    def __init__(self, id, payload):
        self.id, self.payload = id, payload


def test_rrf_surfaces_a_repo_found_by_only_one_arm():
    """Guards the empty-list regression.

    RRF used to be called as `reciprocal_rank_fusion(dense_hits, [], k=60)`. With
    one input it is not a fusion at all — it returns that input reordered by a
    monotonic function of its own rank. The property that proves two arms are
    really merged is that something only ONE arm found still appears.
    """
    dense = [Hit(1, {"full_name": "a/a"}), Hit(2, {"full_name": "b/b"})]
    lexical = [{"id": 3, "payload": {"full_name": "c/c"}}, {"id": 1, "payload": {"full_name": "a/a"}}]

    fused = reciprocal_rank_fusion(dense, lexical, k=60)
    ids = [item["id"] for item in fused]

    assert 3 in ids, "a lexical-only result vanished from the fusion"
    assert 2 in ids, "a dense-only result vanished from the fusion"
    # Found by both arms, so it must outrank anything found by one.
    assert ids[0] == 1, f"repo in both arms should rank first, got {ids}"
    assert len(ids) == len(set(ids)), "the same repo was emitted twice"


def test_empty_lexical_arm_still_returns_dense():
    """The degraded path must still serve, just without fusion's benefit."""
    dense = [Hit(1, {"full_name": "a/a"})]
    assert [i["id"] for i in reciprocal_rank_fusion(dense, [], k=60)] == [1]
    assert reciprocal_rank_fusion([], [], k=60) == []


def test_tsquery_strips_operators():
    """`to_tsquery` parses its own operators, so user text must never reach it raw."""
    assert to_or_tsquery("rust terminal ui") == "rust | terminal | ui"
    # `!`, `&`, `|`, `:*` and parens are syntax to to_tsquery, not search terms.
    assert to_or_tsquery("a & b ! c:* (d)") == "a | b | c | d"
    assert to_or_tsquery("c++ web") == "c | web"
    assert to_or_tsquery("!!!") == "", "a query of pure operators must yield nothing"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall search contract checks passed")
