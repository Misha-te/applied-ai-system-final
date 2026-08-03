"""
Reliability tests for the AI-facing parts of the system.

The other test files cover the deterministic scorer and the guardrails. This one
proves the pieces that make this an *AI* system behave safely and measurably:

  1. LLM-output validation (_clean_prefs) — untrusted JSON from the language
     model is clamped/dropped to safe, in-range preferences before it can steer
     the scorer.
  2. Confidence scoring (confidence_label) — the recommender's self-rating of how
     sure it is maps scores to High/Medium/Low as documented.
  3. The trained model (train_weights) — learned weights are a valid recipe and
     the model clears a minimum in-sample accuracy.
  4. End-to-end reliability (evaluate) — the pipeline recovers each labeled
     listener's known-liked songs at or above a minimum precision.

Thresholds are set conservatively BELOW the numbers we actually observe, so the
suite guards against regressions without being brittle to tiny weight changes.
"""

import json
from pathlib import Path

import pytest

from src.recommender import confidence_label
from src.evaluate import evaluate_all
from src.train_weights import (
    TRAINING_PROFILES,
    build_dataset,
    train_logreg,
    accuracy,
    coeffs_to_weights,
)

LEARNED_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "learned_weights.json"


# --- 1. LLM-output validation: the safety net around the language model ----
# _clean_prefs lives in src/app.py; importing app pulls in Streamlit, which is
# fine in bare mode (it just warns). If Streamlit isn't installed we skip rather
# than fail, so this file still runs in a minimal environment.
try:
    from src.app import _clean_prefs
    HAVE_CLEAN_PREFS = True
except Exception:  # pragma: no cover - only when Streamlit is unavailable
    HAVE_CLEAN_PREFS = False


needs_clean_prefs = pytest.mark.skipif(
    not HAVE_CLEAN_PREFS, reason="src.app (Streamlit) not importable in this environment"
)


@needs_clean_prefs
def test_clean_prefs_clamps_out_of_range_numbers():
    """The model can return nonsense like energy=2.5 or -1; validation must pull
    every numeric field back into the 0..1 the scorer expects."""
    cleaned = _clean_prefs({"energy": 2.5, "valence": -1.0, "danceability": 0.7})
    assert cleaned["energy"] == 1.0
    assert cleaned["valence"] == 0.0
    assert cleaned["danceability"] == 0.7


@needs_clean_prefs
def test_clean_prefs_drops_unknown_categoricals():
    """A genre/mood the catalog doesn't have is dropped, not passed through."""
    cleaned = _clean_prefs({"genre": "polka", "mood": "grumpy"})
    assert "genre" not in cleaned
    assert "mood" not in cleaned


@needs_clean_prefs
def test_clean_prefs_survives_garbage_types():
    """Non-dict input and non-numeric values never raise — they yield {} or are
    skipped, so a malformed model response can't crash the recommendation flow."""
    assert _clean_prefs("not a dict") == {}
    assert _clean_prefs({"energy": "loud", "valence": None}) == {}


# --- 2. Confidence scoring -------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (100.0, "High"), (75.0, "High"),
    (74.9, "Medium"), (50.0, "Medium"),
    (49.9, "Low"), (0.0, "Low"),
])
def test_confidence_label_thresholds(score, expected):
    assert confidence_label(score) == expected


# --- 3. The trained model --------------------------------------------------

def test_learned_weights_file_is_a_valid_recipe():
    """The persisted weights must be non-negative and sum to ~1.0, the shape
    score_song relies on."""
    data = json.loads(LEARNED_WEIGHTS_PATH.read_text(encoding="utf-8"))
    weights = data["weights"]
    assert all(w >= 0.0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_coeffs_to_weights_normalizes_and_floors():
    """Negative coefficients clamp to a small positive floor and the result is a
    normalized distribution — no feature is ever silently dropped."""
    recipe = coeffs_to_weights([-5.0, 0.0, 3.0])
    assert all(w > 0.0 for w in recipe)
    assert sum(recipe) == pytest.approx(1.0, abs=1e-9)


def test_trained_model_clears_minimum_accuracy():
    """Retrain from scratch (deterministic) and require a real minimum accuracy,
    not just 'it ran'. Observed ~0.89; we assert >= 0.80 to allow small drift."""
    from src.recommender import load_songs
    from src.train_weights import CSV_PATH

    X, y = build_dataset(load_songs(str(CSV_PATH)))
    w, b = train_logreg(X, y)
    assert accuracy(X, y, w, b) >= 0.80


def test_training_is_deterministic():
    """Same data in -> same weights out (no randomness), so results reproduce."""
    from src.recommender import load_songs
    from src.train_weights import CSV_PATH

    X, y = build_dataset(load_songs(str(CSV_PATH)))
    first = train_logreg(X, y, epochs=200)
    second = train_logreg(X, y, epochs=200)
    assert first == second


# --- 4. End-to-end reliability ---------------------------------------------

def test_recommender_recovers_known_taste():
    """Across the labeled listeners, the top-5 picks must hit their known likes
    at a mean precision >= 0.5. Observed 0.80; the floor guards against a
    weight change that quietly breaks relevance."""
    report = evaluate_all(k=5)
    assert report["mean_precision"] >= 0.5
    # Every labeled listener should get at least one of their liked songs in top-5.
    assert all(row["hits"] >= 1 for row in report["rows"])


def test_every_profile_is_evaluated():
    report = evaluate_all(k=5)
    assert len(report["rows"]) == len(TRAINING_PROFILES)
