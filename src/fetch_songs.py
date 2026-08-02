"""Build data/songs.csv from real tracks, using the ReccoBeats API only.

Why ReccoBeats (see README "Data" section):
Spotify removed its audio-features endpoints (energy, valence, danceability,
acousticness, tempo) for apps registered after Nov 2024 -- exactly the columns
the recommender scores on (see recommender.py FEATURE_WEIGHTS). ReccoBeats
(api.reccobeats.com) still serves those Spotify-style features, needs NO API key,
and has its own /track/search, so we can build the whole catalog from one source.

What each part supplies:
  ReccoBeats /track/search       -> which songs exist (title, artist, popularity)
  ReccoBeats /audio-features     -> the numbers the recommender scores on
  our SEEDS list                 -> genre (ReccoBeats has no genre field)
  derive_mood(valence, energy)   -> mood (no API has a mood field)

So the human-curated part is just (title, artist, genre) -- the part no API gives
cleanly anyway -- and ReccoBeats supplies the real audio features, replacing the
hand-estimated numbers the CSV used to carry.

Coverage note: ReccoBeats' catalog is strong on Western pop and West-African
afrobeats but thin on East-African bongo/Kenyan artists (Diamond Platnumz,
Nyashinski, Otile Brown do not resolve). Seeds that can't be matched are logged
and skipped, never silently guessed.

Usage (no credentials required):
    python src/fetch_songs.py                 # -> data/songs.csv from SEEDS below
    python src/fetch_songs.py --out data/songs_fetched.csv
    python src/fetch_songs.py --size 30       # widen the search window per seed
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# --- catalog seeds --------------------------------------------------------
# (title, artist, genre). Title drives the ReccoBeats search; artist picks the
# right hit out of the results; genre is our label (ReccoBeats has none). Every
# seed here was verified to resolve to the correct artist -- add your own, and
# the script will tell you at run time if one can't be matched.
SEEDS = [
    ("Anti-Hero",              "Taylor Swift",              "pop"),
    ("Shake It Off",           "Taylor Swift",              "pop"),
    ("24K Magic",              "Bruno Mars",                "funk"),
    ("Talking to the Moon",    "Bruno Mars",                "pop"),
    ("Blinding Lights",        "The Weeknd",                "synthpop"),
    ("Save Your Tears",        "The Weeknd",                "synthpop"),
    ("As It Was",              "Harry Styles",              "pop"),
    ("Levitating",             "Dua Lipa",                  "pop"),
    ("bad guy",                "Billie Eilish",             "pop"),
    ("happier than ever",      "Billie Eilish",             "pop"),
    ("good 4 u",               "Olivia Rodrigo",            "pop rock"),
    ("Smells Like Teen Spirit","Nirvana",                   "rock"),
    ("Kiss Me",                "Sixpence None the Richer",  "indie pop"),
    ("Sura Yako",              "Sauti Sol",                 "afropop"),
    ("Melanin",                "Sauti Sol",                 "afropop"),
    ("One Dance",              "Drake",                     "afrobeats"),
    ("Last Last",              "Burna Boy",                 "afrobeats"),
    ("Calm Down",              "Rema",                      "afrobeats"),
]

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
DEFAULT_SIZE = 20  # how many search hits to scan per seed when matching artist

# CSV columns, in the exact order recommender.load_songs expects.
CSV_FIELDS = [
    "id", "title", "artist", "genre", "mood",
    "energy", "tempo_bpm", "valence", "danceability", "acousticness",
]

RECCOBEATS_API = "https://api.reccobeats.com/v1"
RECCO_BATCH = 40  # ReccoBeats accepts up to 40 ids per audio-features request.


# --- ReccoBeats: discovery ------------------------------------------------
def resolve_seed(title: str, artist: str, size: int) -> Optional[Dict]:
    """Search ReccoBeats for `title` and return the hit matching `artist`.

    ReccoBeats search ranks loosely and is full of covers, so we can't trust the
    first result -- we scan up to `size` hits and take the first whose artist
    list actually contains the seed artist. Returns None (caller logs a miss)
    when the real track isn't in ReccoBeats' catalog.
    """
    resp = requests.get(
        f"{RECCOBEATS_API}/track/search",
        params={"searchText": title, "size": size},
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    for hit in resp.json().get("content", []):
        names = " ".join(a["name"].lower() for a in hit.get("artists", []))
        if artist.lower() in names:
            return {
                "recco_id": hit["id"],
                # Use ReccoBeats' canonical spelling for title/artist.
                "title": hit["trackTitle"],
                "artist": hit["artists"][0]["name"],
            }
    return None


# --- ReccoBeats: audio features -------------------------------------------
def reccobeats_features(recco_ids: List[str]) -> Dict[str, Dict]:
    """Map each ReccoBeats track id -> its audio features, in batches of 40."""
    features: Dict[str, Dict] = {}
    for start in range(0, len(recco_ids), RECCO_BATCH):
        batch = recco_ids[start:start + RECCO_BATCH]
        resp = requests.get(
            f"{RECCOBEATS_API}/audio-features",
            params={"ids": ",".join(batch)},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        for obj in resp.json().get("content", []):
            features[obj["id"]] = obj
        time.sleep(0.2)  # be polite to a free, no-auth API
    return features


# --- derived fields -------------------------------------------------------
def derive_mood(valence: float, energy: float) -> str:
    """Label a mood from valence (positivity) and energy.

    No API has a mood field; the recommender's `mood` column is our own coarse
    read of the two most telling audio features. Kept simple and documented so
    the README can explain exactly how the label is assigned.
    """
    if valence >= 0.6 and energy >= 0.6:
        return "happy"
    if valence >= 0.6 and energy < 0.6:
        return "relaxed"
    if valence < 0.6 and energy >= 0.6:
        return "intense"
    if valence < 0.4 and energy < 0.5:
        return "moody"
    return "chill"


# --- orchestration --------------------------------------------------------
def build_catalog(seeds: List[tuple], size: int) -> List[Dict]:
    """Resolve each seed on ReccoBeats and enrich it with audio features."""
    print(f"Resolving {len(seeds)} seeds on ReccoBeats...")
    resolved: List[Dict] = []
    seen_ids = set()
    for title, artist, genre in seeds:
        hit = resolve_seed(title, artist, size)
        if hit is None:
            print(f"  MISS  {title!r} by {artist} -- not in ReccoBeats, skipping",
                  file=sys.stderr)
            continue
        if hit["recco_id"] in seen_ids:
            continue
        seen_ids.add(hit["recco_id"])
        hit["genre"] = genre
        resolved.append(hit)
        print(f"  ok    {hit['title']!r} by {hit['artist']}")

    if not resolved:
        return []

    print("Fetching audio features...")
    features = reccobeats_features([r["recco_id"] for r in resolved])

    rows: List[Dict] = []
    next_id = 1
    for item in resolved:
        feat = features.get(item["recco_id"])
        if not feat:
            print(f"  ! no audio features for {item['title']!r}, skipping",
                  file=sys.stderr)
            continue
        valence = float(feat["valence"])
        energy = float(feat["energy"])
        rows.append({
            "id": next_id,
            "title": item["title"],
            "artist": item["artist"],
            "genre": item["genre"],
            "mood": derive_mood(valence, energy),
            "energy": round(energy, 3),
            "tempo_bpm": round(float(feat["tempo"]), 1),
            "valence": round(valence, 3),
            "danceability": round(float(feat["danceability"]), 3),
            "acousticness": round(float(feat["acousticness"]), 3),
        })
        next_id += 1
    return rows


FEATURE_COLS = ["mood", "energy", "tempo_bpm", "valence", "danceability", "acousticness"]


def merge_rows(existing: List[Dict], fetched: List[Dict]) -> List[Dict]:
    """Fold fetched rows into an existing catalog, preserving ids and unmatched rows.

    Existing rows keep their id, order, title, artist, and (curated) genre. When a
    fetched song matches an existing one by title+artist, only its audio-feature
    columns are overwritten with the real ReccoBeats values -- so hand-estimated
    numbers get upgraded in place without renumbering anything (main.py and
    sample_profiles reference songs by id). Fetched songs with no match are
    appended with fresh ids. Existing rows ReccoBeats can't provide (e.g. the
    East-African tracks) are left untouched.
    """
    by_key = {(r["title"].lower(), r["artist"].lower()): r for r in existing}
    next_id = max((int(r["id"]) for r in existing), default=0) + 1
    upgraded, added = 0, 0
    for row in fetched:
        match = by_key.get((row["title"].lower(), row["artist"].lower()))
        if match:
            for col in FEATURE_COLS:
                match[col] = row[col]
            upgraded += 1
        else:
            existing.append({**row, "id": next_id})
            next_id += 1
            added += 1
    print(f"Merge: upgraded {upgraded} existing songs, added {added} new ones "
          f"({len(existing)} total).")
    return existing


def read_csv(path: Path) -> List[Dict]:
    """Read an existing catalog CSV back into a list of dict rows (strings kept)."""
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("id")]


def write_csv(rows: List[Dict], out_path: Path) -> None:
    """Write rows to CSV in the schema recommender.load_songs expects."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output CSV path (default data/songs.csv)")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="search hits to scan per seed (default 20)")
    parser.add_argument("--merge", action="store_true",
                        help="merge into the existing --out CSV (upgrade matches, "
                             "append new, keep ids and unresolved rows) instead of "
                             "overwriting it")
    args = parser.parse_args()

    try:
        rows = build_catalog(SEEDS, args.size)
    except requests.HTTPError as exc:
        print(f"ReccoBeats API error: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("No songs resolved; leaving existing CSV untouched.", file=sys.stderr)
        return 1

    if args.merge and args.out.exists():
        rows = merge_rows(read_csv(args.out), rows)

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} songs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
