"""
Learn the recommender's FEATURE_WEIGHTS from data.

This is the project's "fine-tuned / specialized model" component. Instead of
hand-picking how much each audio feature matters, we TRAIN a small logistic-
regression model to predict whether a listener likes a song from that song's
per-feature sub-scores (the same 0-1 sub-scores score_song computes). The
trained model's coefficients tell us how important each feature is; we turn them
into non-negative, normalized weights and write them to data/learned_weights.json,
which recommender.py loads at import time.

Run:
    python -m src.train_weights

No third-party ML libraries are used -- the logistic regression is a few dozen
lines of plain-Python gradient descent, so the whole pipeline is transparent and
fully reproducible: weights start at zero, there is no randomness, so the same
data always yields the same learned weights.
"""

import json
from math import exp
from pathlib import Path

try:  # works as a package (python -m src.train_weights)
    from src.recommender import FEATURE_SPEC, DEFAULT_WEIGHTS, load_songs
except ModuleNotFoundError:  # works as a plain script (python src/train_weights.py)
    from recommender import FEATURE_SPEC, DEFAULT_WEIGHTS, load_songs

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "learned_weights.json"

FEATURE_KEYS = [key for key, _kind, _label in FEATURE_SPEC]

# --- Ground-truth taste profiles (the training labels) --------------------
# Each labeled listener has (a) a full preference spec across all six features
# and (b) a curated set of song ids from data/songs.csv that they are known to
# like. Every other song in the catalog is a negative example for that listener.
# These likes are the supervision signal: the model learns which features
# separate a listener's liked songs from the rest, and how strongly.
TRAINING_PROFILES = [
    {
        "name": "Upbeat pop",
        "prefs": {"genre": "pop", "mood": "happy", "energy": 0.85,
                  "valence": 0.85, "danceability": 0.80, "acousticness": 0.10},
        "likes": [1, 10, 12, 22, 24, 32],
    },
    {
        "name": "Chill study lofi",
        "prefs": {"genre": "lofi", "mood": "chill", "energy": 0.35,
                  "valence": 0.60, "danceability": 0.55, "acousticness": 0.85},
        "likes": [2, 4, 6, 7, 9],
    },
    {
        "name": "Intense rock",
        "prefs": {"genre": "rock", "mood": "intense", "energy": 0.90,
                  "valence": 0.45, "danceability": 0.60, "acousticness": 0.05},
        "likes": [3, 5, 21, 28],
    },
    {
        "name": "Afrobeats / afropop",
        "prefs": {"genre": "afropop", "mood": "happy", "energy": 0.65,
                  "valence": 0.72, "danceability": 0.80, "acousticness": 0.30},
        "likes": [15, 16, 17, 18, 19, 20, 30, 31, 32],
    },
]


def _sub_score(key: str, kind: str, pref, song: dict) -> float:
    """One feature's 0-1 sub-score for a song, matching score_song's math.

    categorical -> 1.0 if the song's value equals the preference, else 0.0.
    numeric     -> 1.0 minus the distance on the shared 0-1 scale, floored at 0.
    """
    if kind == "categorical":
        return 1.0 if str(song.get(key, "")).lower() == str(pref).lower() else 0.0
    target = float(pref)
    value = float(song.get(key, 0.0))
    return max(0.0, min(1.0, 1.0 - abs(value - target)))


def build_dataset(songs: list) -> tuple:
    """Turn (listener, song) pairs into (X, y): sub-score vectors and like labels."""
    X, y = [], []
    for profile in TRAINING_PROFILES:
        prefs = profile["prefs"]
        liked = set(profile["likes"])
        for song in songs:
            X.append([_sub_score(key, kind, prefs[key], song)
                      for key, kind, _label in FEATURE_SPEC])
            y.append(1.0 if song["id"] in liked else 0.0)
    return X, y


def _sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + exp(-z))
    ez = exp(z)
    return ez / (1.0 + ez)


def class_weights(y: list) -> tuple:
    """Per-class example weights that make the two classes count equally.

    Necessary since the catalog grew past a thousand songs: the like-lists still
    hold 24 songs, so ~98% of examples are negatives. Unweighted, the cheapest
    way to cut the loss is to predict "no" for everything -- which is exactly
    what happened, flattening the valence and danceability coefficients to zero.
    Weighting each class by n / (2 * class_count) -- the standard "balanced"
    scheme -- gives the 24 likes the same total pull as the ~1300 non-likes.
    """
    n = len(y)
    n_pos = sum(1 for yi in y if yi >= 0.5)
    n_neg = n - n_pos
    if not n_pos or not n_neg:  # single-class data: nothing to rebalance
        return 1.0, 1.0
    return n / (2.0 * n_pos), n / (2.0 * n_neg)


def train_logreg(X: list, y: list, lr: float = 0.5, epochs: int = 4000,
                 l2: float = 1e-3, balanced: bool = True) -> tuple:
    """Batch-gradient-descent logistic regression. Returns (weights, bias).

    The separate bias absorbs the base like-rate (most songs are negatives), so
    the six feature weights reflect feature importance rather than class balance.
    With `balanced` (the default) each example is additionally scaled by its
    class weight, so a handful of likes can't be drowned out by a catalog full of
    negatives. A small L2 term keeps the weights from blowing up on this small
    dataset. No randomness anywhere: the same inputs always give the same model.
    """
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    n = len(X)
    pos_weight, neg_weight = class_weights(y) if balanced else (1.0, 1.0)
    for _ in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for xi, yi in zip(X, y):
            p = _sigmoid(b + sum(w[j] * xi[j] for j in range(n_features)))
            err = (p - yi) * (pos_weight if yi >= 0.5 else neg_weight)
            for j in range(n_features):
                grad_w[j] += err * xi[j]
            grad_b += err
        for j in range(n_features):
            w[j] -= lr * (grad_w[j] / n + l2 * w[j])
        b -= lr * (grad_b / n)
    return w, b


def coeffs_to_weights(w: list) -> list:
    """Turn raw model coefficients into a usable weight recipe.

    A weight only makes sense as a non-negative "how much this feature counts",
    so clamp negative coefficients to 0. A small floor keeps every feature in
    play (no feature is silently dropped), then we normalize so the weights sum
    to 1.0 -- the same shape score_song expects.
    """
    floor = 0.01
    clamped = [max(0.0, wj) + floor for wj in w]
    total = sum(clamped)
    return [c / total for c in clamped]


def accuracy(X: list, y: list, w: list, b: float) -> float:
    """Share of examples the trained model classifies correctly (threshold 0.5).

    Read this next to balanced_accuracy: with ~98% of the catalog labeled "not
    liked", answering "no" to everything already scores ~98% here, so plain
    accuracy stopped being informative once the catalog grew.
    """
    correct = 0
    for xi, yi in zip(X, y):
        p = _sigmoid(b + sum(w[j] * xi[j] for j in range(len(w))))
        if (1.0 if p >= 0.5 else 0.0) == yi:
            correct += 1
    return correct / len(X)


def balanced_accuracy(X: list, y: list, w: list, b: float) -> float:
    """Mean of the two per-class hit rates: (likes right + non-likes right) / 2.

    This is the honest headline number for an imbalanced dataset -- an
    always-say-no model scores 0.50 here, not 0.98.
    """
    hits = {1.0: 0, 0.0: 0}
    totals = {1.0: 0, 0.0: 0}
    for xi, yi in zip(X, y):
        label = 1.0 if yi >= 0.5 else 0.0
        p = _sigmoid(b + sum(w[j] * xi[j] for j in range(len(w))))
        totals[label] += 1
        if (1.0 if p >= 0.5 else 0.0) == label:
            hits[label] += 1
    rates = [hits[c] / totals[c] for c in (1.0, 0.0) if totals[c]]
    return sum(rates) / len(rates) if rates else 0.0


def main() -> None:
    songs = load_songs(str(CSV_PATH))
    X, y = build_dataset(songs)
    w, b = train_logreg(X, y)
    weights = coeffs_to_weights(w)
    weight_map = {key: round(weights[i], 4) for i, key in enumerate(FEATURE_KEYS)}
    acc = accuracy(X, y, w, b)
    bal_acc = balanced_accuracy(X, y, w, b)

    OUT_PATH.write_text(json.dumps({
        "weights": weight_map,
        "raw_coefficients": {key: round(w[i], 4) for i, key in enumerate(FEATURE_KEYS)},
        "bias": round(b, 4),
        "train_accuracy": round(acc, 4),
        "train_balanced_accuracy": round(bal_acc, 4),
        "n_examples": len(X),
        "n_listeners": len(TRAINING_PROFILES),
        "note": "Learned by src/train_weights.py logistic regression. Do not hand-edit; re-run the trainer instead.",
    }, indent=2) + "\n", encoding="utf-8")

    positives = int(sum(y))
    majority = 1.0 - positives / len(X)  # what "always say no" would score
    print(f"Trained on {len(X)} (listener, song) examples "
          f"from {len(TRAINING_PROFILES)} listeners ({positives} likes, {len(X) - positives} non-likes).")
    print(f"Balanced accuracy: {bal_acc:.1%}   (likes and non-likes weighted equally)")
    print(f"Plain accuracy:    {acc:.1%}   (always-say-no baseline: {majority:.1%})\n")
    print(f"{'feature':<14}{'hand-tuned':>12}{'learned':>10}")
    print("-" * 36)
    for key in FEATURE_KEYS:
        print(f"{key:<14}{DEFAULT_WEIGHTS[key]:>12.2f}{weight_map[key]:>10.2f}")
    print(f"\nWrote learned weights to {OUT_PATH}")
    print("recommender.py will pick these up automatically on next import.")


if __name__ == "__main__":
    main()
