import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# --- Scoring recipe: LEARNED feature weights ------------------------------
# Each feature contributes a 0-1 sub-score; the weight says how much that
# feature counts toward a song's final 0-100% score. Instead of hand-picking
# these weights, we LEARN them from data: src/train_weights.py trains a small
# logistic-regression model to predict whether a listener likes a song from its
# per-feature sub-scores, and the model's (non-negative, normalized) coefficients
# become the weights used here. This is the project's "fine-tuned / specialized
# model" component -- the recipe is trained, not guessed.
#
# Retrain with:  python -m src.train_weights   (refreshes data/learned_weights.json)
# If that file is absent we fall back to the original hand-tuned recipe below, so
# the recommender always runs out of the box.

# Canonical feature list: (key, kind, label shown in reasons). Each feature's
# weight is filled in from the learned file (or DEFAULT_WEIGHTS) at import time.
FEATURE_SPEC = [
    ("genre",        "categorical", "Genre"),
    ("energy",       "numeric",     "Energy"),
    ("valence",      "numeric",     "Valence"),
    ("danceability", "numeric",     "Danceability"),
    ("mood",         "categorical", "Mood"),
    ("acousticness", "numeric",     "Acousticness"),
]

# Original hand-tuned recipe, used only as a fallback when no learned weights
# file is present. Sums to 1.00.
DEFAULT_WEIGHTS = {
    "genre":        0.30,
    "energy":       0.20,
    "valence":      0.15,
    "danceability": 0.15,
    "mood":         0.15,
    "acousticness": 0.05,
}

LEARNED_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "learned_weights.json"


def _load_feature_weights() -> List[Tuple[str, float, str, str]]:
    """Return FEATURE_WEIGHTS as [(key, weight, kind, label)].

    Uses the trained weights in data/learned_weights.json when available, and
    the hand-tuned DEFAULT_WEIGHTS otherwise. Only known feature keys are read
    from the file, so a malformed or partial file degrades gracefully.
    """
    weights = dict(DEFAULT_WEIGHTS)
    try:
        with open(LEARNED_WEIGHTS_PATH, encoding="utf-8") as f:
            learned = json.load(f).get("weights", {})
        for key in weights:
            if key in learned:
                weights[key] = float(learned[key])
    except (OSError, ValueError, KeyError, AttributeError):
        pass  # keep hand-tuned defaults if the file is missing or unreadable
    return [(key, weights[key], kind, label) for key, kind, label in FEATURE_SPEC]


FEATURE_WEIGHTS = _load_feature_weights()

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # World region the artist is from ("east africa", "latin america", ...).
    # Descriptive only -- nothing is scored on it -- and defaulted so older
    # catalogs and hand-built Song objects still construct.
    region: str = ""

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.

    Design (see README "How The System Works"): a profile stores basic user
    info plus the user's *average taste* across the three numeric audio
    features we use for recommending. Songs the user plays repeatedly are
    tracked separately so they can count more when we build that average.
    """
    # --- basic user info ---
    name: str = "Guest"

    # --- average taste, from the songs this user likes/listens to ---
    avg_energy: float = 0.5
    avg_valence: float = 0.5
    avg_danceability: float = 0.5

    # --- songs the user plays repeatedly (ids from songs.csv) ---
    replayed_song_ids: List[int] = field(default_factory=list)

    # --- optional categorical taste the user can declare directly ---
    # Used by profile_to_prefs to add genre/mood preferences on top of the
    # numeric averages. Left blank for a profile learned purely from history.
    favorite_genre: str = ""
    favorite_mood: str = ""

NUMERIC_FIELDS = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into a list of dicts, numeric fields as floats."""
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("id"):  # skip blank trailing lines
                continue
            song = dict(row)
            song["id"] = int(row["id"])
            for field_name in NUMERIC_FIELDS:
                if song.get(field_name) not in (None, ""):
                    song[field_name] = float(song[field_name])
            songs.append(song)
    return songs

def to_song(row: Dict) -> Song:
    """Turn one loaded song dict (from load_songs) into a Song object.

    The catalog is loaded as dicts because that is what the scoring pipeline
    consumes, but build_profile works on Song objects. This converter is the
    single bridge between the two, so a profile can be built from the very same
    songs the recommender ranks.
    """
    return Song(
        id=int(row["id"]),
        title=row["title"],
        artist=row["artist"],
        genre=row["genre"],
        mood=row["mood"],
        energy=float(row["energy"]),
        tempo_bpm=float(row["tempo_bpm"]),
        valence=float(row["valence"]),
        danceability=float(row["danceability"]),
        acousticness=float(row["acousticness"]),
        region=row.get("region", ""),
    )

def confidence_label(score: float) -> str:
    """Turn a 0-100 match score into a plain-language confidence rating.

    This is how the recommender "says how sure it is" about a pick: a high score
    means many weighted features lined up, a low score means few did. Thresholds
    are deliberately simple and are exercised by tests/test_ai_reliability.py.
    """
    if score >= 75.0:
        return "High"
    if score >= 50.0:
        return "Medium"
    return "Low"


def _numeric_reason(label: str, value: float, target: float, closeness: float) -> str:
    """Turn a numeric feature's closeness into a human-readable reason."""
    if closeness >= 0.90:
        return f"{label} is very close to your preferred level"
    if closeness >= 0.75:
        return f"{label} is close to your preferred level"
    direction = "higher" if value > target else "lower"
    return f"{label} is {direction} than your preferred level"


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user_prefs, returning (percent_score, reasons)."""
    weighted_sum = 0.0
    used_weight = 0.0
    reasons: List[str] = []

    for key, weight, kind, label in FEATURE_WEIGHTS:
        # Only score features the user actually expressed a preference for.
        pref = user_prefs.get(key)
        if pref is None or pref == "":
            continue
        used_weight += weight

        if kind == "categorical":
            matched = str(song.get(key, "")).lower() == str(pref).lower()
            sub = 1.0 if matched else 0.0
            if matched:
                reasons.append(f"{label} matches your preference for {pref}")
            else:
                reasons.append(f"{label} ({song.get(key)}) differs from your preferred {pref}")
        else:  # numeric: both values are on the same 0-1 scale
            target = float(pref)
            value = float(song.get(key, 0.0))
            sub = max(0.0, min(1.0, 1.0 - abs(value - target)))
            reasons.append(_numeric_reason(label, value, target, sub))

        weighted_sum += weight * sub

    if used_weight == 0.0:
        return 0.0, ["No matching preferences were provided to score this song."]

    # Normalize by the weight actually used so the result is a true 0-100%
    # even when the user only specified some of the features.
    score = max(0.0, min(100.0, (weighted_sum / used_weight) * 100.0))
    return score, reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    artist_penalty: float = 20.0,
    genre_penalty: float = 10.0,
) -> List[Tuple[Dict, float, List[str]]]:
    """Score, rank high->low with a diversity penalty for repeats, return top-k (song, score, reasons)."""
    # k must be a real integer -- bool is an int subclass, so reject it too.
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    # Base score for every song (never mutate the catalog itself).
    pool: List[Dict] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        pool.append({"song": song, "base": score, "reasons": reasons})

    # Greedy diversity-aware ranking: build the list one slot at a time. Before
    # picking the next song, penalize any candidate whose artist or genre is
    # ALREADY in the chosen list, so the top results don't pile up on one
    # artist/genre. Set both penalties to 0.0 to get plain score ordering.
    chosen: List[Dict] = []
    while pool and len(chosen) < k:
        best = None
        best_adjusted = -1.0
        best_penalty = 0.0
        best_repeats: List[str] = []

        for item in pool:
            penalty = 0.0
            repeats: List[str] = []
            for picked in chosen:
                if picked["song"].get("artist") == item["song"].get("artist"):
                    penalty += artist_penalty
                    repeats.append(f"artist {item['song'].get('artist')}")
                if picked["song"].get("genre") == item["song"].get("genre"):
                    penalty += genre_penalty
                    repeats.append(f"genre {item['song'].get('genre')}")
            adjusted = max(0.0, item["base"] - penalty)
            # Highest adjusted score wins; ties fall back to the raw match score.
            if adjusted > best_adjusted or (
                adjusted == best_adjusted and best is not None and item["base"] > best["base"]
            ):
                best, best_adjusted, best_penalty = item, adjusted, penalty
                best_repeats = repeats

        pool.remove(best)
        reasons = list(best["reasons"])
        if best_penalty > 0:
            shared = ", ".join(sorted(set(best_repeats)))
            reasons.append(
                f"Diversity penalty: -{best_penalty:.0f} because its {shared} already appears above"
            )
        chosen.append({"song": best["song"], "score": best_adjusted, "reasons": reasons})

    return [(c["song"], c["score"], c["reasons"]) for c in chosen]

def build_profile(
    name: str,
    songs: List[Song],
    liked_ids: List[int],
    replayed_ids: Optional[List[int]] = None,
) -> UserProfile:
    """
    Build a UserProfile by averaging the audio features of the songs a user
    likes. Songs the user replays repeatedly are counted twice, so they pull
    the average toward the user's strongest preferences.
    """
    replayed_ids = replayed_ids or []
    songs_by_id = {song.id: song for song in songs}

    # Build a weighted list: a replayed song appears twice.
    listened: List[Song] = []
    for song_id in liked_ids:
        song = songs_by_id.get(song_id)
        if song is None:
            continue
        weight = 2 if song_id in replayed_ids else 1
        listened.extend([song] * weight)

    if not listened:
        return UserProfile(name=name, replayed_song_ids=list(replayed_ids))

    count = len(listened)
    return UserProfile(
        name=name,
        avg_energy=round(sum(s.energy for s in listened) / count, 3),
        avg_valence=round(sum(s.valence for s in listened) / count, 3),
        avg_danceability=round(sum(s.danceability for s in listened) / count, 3),
        replayed_song_ids=list(replayed_ids),
    )

def profile_to_prefs(profile: UserProfile) -> Dict:
    """Translate a UserProfile into the preference dict score_song consumes.

    This is what wires the OOP side (a profile built from someone's listening
    history) into the functional recommender: build_profile -> profile_to_prefs
    -> recommend_songs. Only fields the user actually expressed are included;
    score_song already skips anything left blank, and normalizes by the weight
    of the features that ARE present, so a numbers-only profile still scores 0-100%.
    """
    prefs: Dict = {
        "energy": profile.avg_energy,
        "valence": profile.avg_valence,
        "danceability": profile.avg_danceability,
    }
    # Categorical taste is optional -- a history-built profile may only know the
    # user's average audio features, not a declared favorite genre/mood.
    if profile.favorite_genre:
        prefs["genre"] = profile.favorite_genre
    if profile.favorite_mood:
        prefs["mood"] = profile.favorite_mood
    return prefs

def sample_profiles() -> List[UserProfile]:
    """
    A few example users with different tastes, for testing the recommender.
    The average values below are hand-set to represent each taste; normally
    you would compute them from a user's listening history with build_profile.
    """
    return [
        # Pop fan (Taylor Swift, Bruno Mars): upbeat, danceable, positive.
        UserProfile(
            name="Amina (pop fan)",
            avg_energy=0.76,
            avg_valence=0.77,
            avg_danceability=0.75,
            replayed_song_ids=[12, 13],  # Shake It Off, 24K Magic
        ),
        # Afropop / bongo fan (Diamond Platnumz, Sauti Sol): warm and danceable.
        UserProfile(
            name="Baraka (afropop fan)",
            avg_energy=0.66,
            avg_valence=0.72,
            avg_danceability=0.79,
            replayed_song_ids=[15, 17],  # Jeje, Sura Yako
        ),
        # Study / chill listener (lofi, ambient): calm, low energy.
        UserProfile(
            name="Chris (chill listener)",
            avg_energy=0.36,
            avg_valence=0.60,
            avg_danceability=0.55,
            replayed_song_ids=[9],  # Focus Flow
        ),
    ]
