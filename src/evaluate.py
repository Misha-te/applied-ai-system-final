"""
Measure how RELIABLE the recommender is, not just whether it runs.

The unit tests in tests/ prove individual functions behave; this harness answers
a different question: when a listener with a known taste asks for songs, does the
recommender actually surface the songs that listener is known to like?

We reuse the four labeled taste profiles from src/train_weights.py (each has a
full preference spec plus a curated set of liked song ids). For each listener we
rank the catalog and check where their liked songs land:

  precision@k = share of the top-k picks that the listener actually likes
  recall@k    = share of the listener's liked songs that made the top-k
  confidence  = the recommender's own top-1 score, via confidence_label()

Two pools, and the difference matters
-------------------------------------
The likes were curated when the catalog was 32 songs; the catalog is now in the
hundreds, pulled from the ReccoBeats API (see src/fetch_songs.py). Precision is a
CLOSED-WORLD metric -- anything not on the like-list counts as a miss -- so a
genuinely great unlabeled afrobeats pick scores exactly like a bad one. That is a
gap in the LABELS, not a regression in the recommender, so we report both:

  labeled pool -> ids 1..LABELED_CATALOG_MAX_ID, the slice the labels actually
                  cover. This is the honest reliability guard, and the number the
                  test in tests/test_ai_reliability.py asserts on.
  full catalog -> every song. Reported for context only; it drops as the catalog
                  grows past the labels, and reads as "how much unlabeled
                  material now outranks the labeled likes".

Run:
    python -m src.evaluate

This is an IN-SAMPLE check (the same profiles that trained the weights), so read
it as "does the pipeline recover known preferences", not as generalization to new
users. It is deterministic: no randomness, so the numbers are reproducible.
"""

from pathlib import Path
from typing import Dict, List

try:  # works as a package (python -m src.evaluate)
    from src.recommender import (
        load_songs,
        recommend_songs,
        confidence_label,
    )
    from src.train_weights import TRAINING_PROFILES
except ModuleNotFoundError:  # works as a plain script (python src/evaluate.py)
    from recommender import load_songs, recommend_songs, confidence_label
    from train_weights import TRAINING_PROFILES

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

# The catalog held 32 songs when the like-lists in TRAINING_PROFILES were
# curated, and fetch_songs.py appends new songs with fresh ids rather than
# renumbering. So ids 1..32 are exactly the slice the labels cover.
LABELED_CATALOG_MAX_ID = 32


def evaluate_profile(profile: Dict, songs: List[Dict], k: int = 5) -> Dict:
    """Rank the catalog for one labeled listener and score the top-k against
    their known likes. Diversity penalties are off here so we measure pure
    relevance, not the spread-the-artists reshuffle."""
    liked = set(profile["likes"])
    ranked = recommend_songs(
        profile["prefs"], songs, k=k, artist_penalty=0.0, genre_penalty=0.0
    )

    top_ids = [song["id"] for song, _score, _reasons in ranked]
    hits = sum(1 for song_id in top_ids if song_id in liked)
    top_score = ranked[0][1] if ranked else 0.0

    return {
        "name": profile["name"],
        "k": k,
        "n_likes": len(liked),
        "hits": hits,
        "precision": hits / k if k else 0.0,
        "recall": hits / len(liked) if liked else 0.0,
        "top_score": top_score,
        "confidence": confidence_label(top_score),
    }


def labeled_pool(songs: List[Dict]) -> List[Dict]:
    """The slice of the catalog the curated like-lists actually cover."""
    return [s for s in songs if int(s["id"]) <= LABELED_CATALOG_MAX_ID]


def evaluate_all(k: int = 5, pool: str = "labeled") -> Dict:
    """Evaluate every labeled profile and return per-profile rows plus the means
    across profiles. Callable from tests so reliability can be asserted, not just
    printed.

    pool="labeled" (default) scores against ids 1..LABELED_CATALOG_MAX_ID, the
    only songs the like-lists judge; pool="full" scores against the whole
    catalog, where unlabeled songs count as misses by construction.
    """
    songs = load_songs(str(CSV_PATH))
    if pool == "labeled":
        songs = labeled_pool(songs)
    elif pool != "full":
        raise ValueError(f"pool must be 'labeled' or 'full', got {pool!r}")
    rows = [evaluate_profile(p, songs, k=k) for p in TRAINING_PROFILES]
    n = len(rows) or 1
    return {
        "k": k,
        "pool": pool,
        "n_songs": len(songs),
        "rows": rows,
        "mean_precision": sum(r["precision"] for r in rows) / n,
        "mean_recall": sum(r["recall"] for r in rows) / n,
        "mean_top_score": sum(r["top_score"] for r in rows) / n,
    }


def _print_report(report: Dict, title: str) -> None:
    k = report["k"]
    print(f"{title} — precision/recall @ top-{k}, across "
          f"{len(report['rows'])} labeled listeners ({report['n_songs']} songs)\n")
    header = f"{'listener':<22}{'likes':>6}{'hits':>6}{'prec':>7}{'recall':>8}{'top%':>7}  conf"
    print(header)
    print("-" * len(header))
    for r in report["rows"]:
        print(f"{r['name']:<22}{r['n_likes']:>6}{r['hits']:>6}"
              f"{r['precision']:>7.2f}{r['recall']:>8.2f}{r['top_score']:>7.1f}  {r['confidence']}")
    print("-" * len(header))
    print(f"{'MEAN':<22}{'':>6}{'':>6}"
          f"{report['mean_precision']:>7.2f}{report['mean_recall']:>8.2f}"
          f"{report['mean_top_score']:>7.1f}")


def main() -> None:
    labeled = evaluate_all(k=5, pool="labeled")
    _print_report(labeled, "Recommender reliability (labeled pool)")
    print("\nNote: in-sample check on the profiles used to learn the weights — "
          "reads as 'recovers known taste', not new-user generalization.\n")

    full = evaluate_all(k=5, pool="full")
    print("=" * 62 + "\n")
    _print_report(full, "Same profiles, ranked against the FULL catalog")
    print("\nNote: context only, not a regression. The like-lists were curated "
          f"when the catalog was {LABELED_CATALOG_MAX_ID} songs, so every one of "
          f"the other {full['n_songs'] - labeled['n_songs']} songs counts as a "
          "miss even when it is\na better match. Expanding the labels to cover "
          "the fetched catalog is the fix; see README.")


if __name__ == "__main__":
    main()
