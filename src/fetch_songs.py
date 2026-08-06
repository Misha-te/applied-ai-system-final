"""Build data/songs.csv from real tracks, using the ReccoBeats API only.

Why ReccoBeats (see README "Data" section):
Spotify removed its audio-features endpoints (energy, valence, danceability,
acousticness, tempo) for apps registered after Nov 2024 -- exactly the columns
the recommender scores on (see recommender.py FEATURE_WEIGHTS). ReccoBeats
(api.reccobeats.com) still serves those Spotify-style features, needs NO API key,
and has its own search endpoints, so we can build the whole catalog from one source.

Two ways to build the catalog
-----------------------------
1. Track mode (--mode tracks): resolve the hand-listed SEEDS one song at a time
   via /track/search. Precise, but it only ever yields as many songs as you type.
2. Artist mode (--mode artists, the DEFAULT): walk the ARTIST_ROSTER below --
   189 artists tagged with a genre and a world region -- pull each artist's
   tracks from /artist/{id}/track, and round-robin across artists until we hit
   --target songs. This is how the catalog grew past a thousand songs with real
   coverage outside the Anglo-American pop bubble.

What each part supplies:
  ReccoBeats /artist/search      -> which artists exist (and their ReccoBeats ids)
  ReccoBeats /artist/{id}/track  -> which songs exist (title, artist, popularity)
  ReccoBeats /track/search       -> same, for one named song (track mode)
  ReccoBeats /audio-features     -> the numbers the recommender scores on
  our ARTIST_ROSTER / SEEDS      -> genre + region (ReccoBeats has neither field)
  derive_mood(valence, energy)   -> mood (no API has a mood field)

So the human-curated part is just (artist, genre, region) -- the part no API
gives cleanly anyway -- and ReccoBeats supplies the real titles and real audio
features, so no number in the CSV is invented.

Coverage note: an artist is only kept when /artist/search returns an exact
name match AND the artist is the *primary* credit on the track, so a featured
verse on someone else's song never gets mislabeled with the wrong region.
Artists ReccoBeats doesn't carry are logged as MISS and skipped, never guessed.

Usage (no credentials required):
    python src/fetch_songs.py --merge                  # +1000 songs, all regions
    python src/fetch_songs.py --target 300 --merge     # smaller pull
    python src/fetch_songs.py --region "east africa" --target 100 --merge
    python src/fetch_songs.py --mode tracks --merge    # the old per-song SEEDS path
    python src/fetch_songs.py --out data/songs_new.csv # write elsewhere, no merge
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import unicodedata
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

# --- artist roster --------------------------------------------------------
# (artist, genre, region). Artist mode expands each of these into up to
# --per-artist real tracks. `region` is our own label -- it is what makes the
# catalog listen like the world instead of like one radio market -- and `genre`
# is kept to a small controlled vocabulary because src/app.py shows the full
# genre list to the LLM when it translates a chat answer into preferences.
ARTIST_ROSTER = [
    # --- East Africa ------------------------------------------------------
    ("Diamond Platnumz",        "bongo",           "east africa"),
    ("Harmonize",               "bongo",           "east africa"),
    ("Rayvanny",                "bongo",           "east africa"),
    ("Alikiba",                 "bongo",           "east africa"),
    ("Zuchu",                   "bongo",           "east africa"),
    ("Mbosso",                  "bongo",           "east africa"),
    ("Nandy",                   "bongo",           "east africa"),
    ("Marioo",                  "bongo",           "east africa"),
    ("Jux",                     "bongo",           "east africa"),
    ("Sauti Sol",               "afropop",         "east africa"),
    ("Nyashinski",              "afropop",         "east africa"),
    ("Otile Brown",             "afropop",         "east africa"),
    ("Bien",                    "afropop",         "east africa"),
    ("Nviiri the Storyteller",  "afropop",         "east africa"),
    ("Khaligraph Jones",        "hip hop",         "east africa"),
    ("Willy Paul",              "afropop",         "east africa"),
    ("Eddy Kenzo",              "afropop",         "east africa"),
    ("Sheebah Karungi",         "afropop",         "east africa"),
    ("Bebe Cool",               "afropop",         "east africa"),
    ("Teddy Afro",              "ethio pop",       "east africa"),
    ("Aster Aweke",             "ethio pop",       "east africa"),
    ("Mulatu Astatke",          "ethio jazz",      "east africa"),
    # --- West Africa ------------------------------------------------------
    ("Burna Boy",               "afrobeats",       "west africa"),
    ("Wizkid",                  "afrobeats",       "west africa"),
    ("Davido",                  "afrobeats",       "west africa"),
    ("Rema",                    "afrobeats",       "west africa"),
    ("Asake",                   "afrobeats",       "west africa"),
    ("Ayra Starr",              "afrobeats",       "west africa"),
    ("Tems",                    "afrobeats",       "west africa"),
    ("Fireboy DML",             "afrobeats",       "west africa"),
    ("Omah Lay",                "afrobeats",       "west africa"),
    ("Tiwa Savage",             "afrobeats",       "west africa"),
    ("CKay",                    "afrobeats",       "west africa"),
    ("Joeboy",                  "afrobeats",       "west africa"),
    ("Yemi Alade",              "afropop",         "west africa"),
    ("Mr Eazi",                 "afrobeats",       "west africa"),
    ("Flavour",                 "highlife",        "west africa"),
    ("P-Square",                "afropop",         "west africa"),
    ("Fela Kuti",               "afrobeat",        "west africa"),
    ("Sarkodie",                "hip hop",         "west africa"),
    ("Shatta Wale",             "dancehall",       "west africa"),
    ("Stonebwoy",               "dancehall",       "west africa"),
    ("Black Sherif",            "afrobeats",       "west africa"),
    ("Amaarae",                 "alte",            "west africa"),
    ("Youssou N'Dour",          "mbalax",          "west africa"),
    ("Salif Keita",             "mande",           "west africa"),
    ("Angelique Kidjo",         "afropop",         "west africa"),
    ("Tinariwen",               "desert blues",    "west africa"),
    # --- Central Africa ---------------------------------------------------
    ("Fally Ipupa",             "rumba",           "central africa"),
    ("Koffi Olomide",           "rumba",           "central africa"),
    ("Papa Wemba",              "rumba",           "central africa"),
    ("Innoss'B",                "rumba",           "central africa"),
    ("Franco",                  "rumba",           "central africa"),
    # --- Southern Africa --------------------------------------------------
    ("Master KG",               "amapiano",        "southern africa"),
    ("Kabza De Small",          "amapiano",        "southern africa"),
    ("DJ Maphorisa",            "amapiano",        "southern africa"),
    ("Focalistic",              "amapiano",        "southern africa"),
    ("Tyla",                    "amapiano",        "southern africa"),
    ("Sho Madjozi",             "gqom",            "southern africa"),
    ("Nasty C",                 "hip hop",         "southern africa"),
    ("Black Coffee",            "house",           "southern africa"),
    ("Ladysmith Black Mambazo", "isicathamiya",    "southern africa"),
    ("Miriam Makeba",           "afropop",         "southern africa"),
    ("Zahara",                  "afro soul",       "southern africa"),
    # --- North Africa & Middle East ---------------------------------------
    ("Amr Diab",                "arabic pop",      "north africa & middle east"),
    ("Nancy Ajram",             "arabic pop",      "north africa & middle east"),
    ("Elissa",                  "arabic pop",      "north africa & middle east"),
    ("Tamer Hosny",             "arabic pop",      "north africa & middle east"),
    ("Saad Lamjarred",          "arabic pop",      "north africa & middle east"),
    ("Fairuz",                  "arabic classical","north africa & middle east"),
    ("Umm Kulthum",             "arabic classical","north africa & middle east"),
    ("Khaled",                  "rai",             "north africa & middle east"),
    ("Wegz",                    "hip hop",         "north africa & middle east"),
    ("Mohamed Ramadan",         "arabic pop",      "north africa & middle east"),
    ("Tarkan",                  "turkish pop",     "north africa & middle east"),
    ("Sezen Aksu",              "turkish pop",     "north africa & middle east"),
    # --- Latin America ----------------------------------------------------
    ("Bad Bunny",               "reggaeton",       "latin america"),
    ("J Balvin",                "reggaeton",       "latin america"),
    ("Karol G",                 "reggaeton",       "latin america"),
    ("Ozuna",                   "reggaeton",       "latin america"),
    ("Maluma",                  "reggaeton",       "latin america"),
    ("Shakira",                 "latin pop",       "latin america"),
    ("Luis Miguel",             "latin pop",       "latin america"),
    ("Juanes",                  "latin rock",      "latin america"),
    ("Soda Stereo",             "latin rock",      "latin america"),
    ("Natalia Lafourcade",      "latin folk",      "latin america"),
    ("Peso Pluma",              "regional mexican","latin america"),
    ("Grupo Frontera",          "regional mexican","latin america"),
    ("Vicente Fernandez",       "ranchera",        "latin america"),
    ("Carlos Vives",            "vallenato",       "latin america"),
    ("Ruben Blades",            "salsa",           "latin america"),
    ("Marc Anthony",            "salsa",           "latin america"),
    ("Celia Cruz",              "salsa",           "latin america"),
    ("Anitta",                  "brazilian funk",  "latin america"),
    ("Caetano Veloso",          "mpb",             "latin america"),
    ("Gilberto Gil",            "mpb",             "latin america"),
    ("Jorge Ben Jor",           "samba",           "latin america"),
    ("Joao Gilberto",           "bossa nova",      "latin america"),
    # --- Caribbean --------------------------------------------------------
    ("Bob Marley",              "reggae",          "caribbean"),
    ("Burning Spear",           "reggae",          "caribbean"),
    ("Chronixx",                "reggae",          "caribbean"),
    ("Koffee",                  "reggae",          "caribbean"),
    ("Damian Marley",           "reggae",          "caribbean"),
    ("Sean Paul",               "dancehall",       "caribbean"),
    ("Buju Banton",             "dancehall",       "caribbean"),
    ("Shenseea",                "dancehall",       "caribbean"),
    ("Machel Montano",          "soca",            "caribbean"),
    ("Kes",                     "soca",            "caribbean"),
    # --- South Asia -------------------------------------------------------
    ("Arijit Singh",            "bollywood",       "south asia"),
    ("Shreya Ghoshal",          "bollywood",       "south asia"),
    ("A.R. Rahman",             "bollywood",       "south asia"),
    ("Lata Mangeshkar",         "bollywood",       "south asia"),
    ("Neha Kakkar",             "bollywood",       "south asia"),
    ("Anirudh Ravichander",     "tamil pop",       "south asia"),
    ("Sid Sriram",              "tamil pop",       "south asia"),
    ("Diljit Dosanjh",          "punjabi",         "south asia"),
    ("Sidhu Moose Wala",        "punjabi",         "south asia"),
    ("AP Dhillon",              "punjabi",         "south asia"),
    ("Atif Aslam",              "pakistani pop",   "south asia"),
    ("Nusrat Fateh Ali Khan",   "qawwali",         "south asia"),
    ("Ravi Shankar",            "hindustani",      "south asia"),
    # --- East Asia --------------------------------------------------------
    ("BTS",                     "k-pop",           "east asia"),
    ("BLACKPINK",               "k-pop",           "east asia"),
    ("IU",                      "k-pop",           "east asia"),
    ("Stray Kids",              "k-pop",           "east asia"),
    ("NewJeans",                "k-pop",           "east asia"),
    ("TWICE",                   "k-pop",           "east asia"),
    ("SEVENTEEN",               "k-pop",           "east asia"),
    ("Jay Chou",                "mandopop",        "east asia"),
    ("Teresa Teng",             "mandopop",        "east asia"),
    ("JJ Lin",                  "mandopop",        "east asia"),
    ("Utada Hikaru",            "j-pop",           "east asia"),
    ("Kenshi Yonezu",           "j-pop",           "east asia"),
    ("YOASOBI",                 "j-pop",           "east asia"),
    ("Fujii Kaze",              "j-pop",           "east asia"),
    ("Babymetal",               "j-metal",         "east asia"),
    # --- Southeast Asia ---------------------------------------------------
    ("Rich Brian",              "hip hop",         "southeast asia"),
    ("NIKI",                    "indie pop",       "southeast asia"),
    ("Tulus",                   "indo pop",        "southeast asia"),
    ("Raisa",                   "indo pop",        "southeast asia"),
    ("Agnez Mo",                "indo pop",        "southeast asia"),
    ("Ben&Ben",                 "opm",             "southeast asia"),
    ("Moira Dela Torre",        "opm",             "southeast asia"),
    ("Sarah Geronimo",          "opm",             "southeast asia"),
    ("Phum Viphurit",           "indie pop",       "southeast asia"),
    ("Son Tung M-TP",           "v-pop",           "southeast asia"),
    ("Hoang Thuy Linh",         "v-pop",           "southeast asia"),
    # --- Europe -----------------------------------------------------------
    ("Adele",                   "pop",             "europe"),
    ("Ed Sheeran",              "pop",             "europe"),
    ("Coldplay",                "rock",            "europe"),
    ("Dua Lipa",                "pop",             "europe"),
    ("Stromae",                 "french pop",      "europe"),
    ("Aya Nakamura",            "french pop",      "europe"),
    ("Edith Piaf",              "chanson",         "europe"),
    ("Daft Punk",               "electronic",      "europe"),
    ("David Guetta",            "electronic",      "europe"),
    ("Rosalia",                 "flamenco pop",    "europe"),
    ("Maneskin",                "rock",            "europe"),
    ("Rammstein",               "metal",           "europe"),
    ("Kraftwerk",               "electronic",      "europe"),
    ("ABBA",                    "pop",             "europe"),
    ("Bjork",                   "art pop",         "europe"),
    ("Sigur Ros",               "post rock",       "europe"),
    # --- North America ----------------------------------------------------
    ("Taylor Swift",            "pop",             "north america"),
    ("Beyonce",                 "r&b",             "north america"),
    ("The Weeknd",              "synthpop",        "north america"),
    ("Bruno Mars",              "funk",            "north america"),
    ("Billie Eilish",           "pop",             "north america"),
    ("Olivia Rodrigo",          "pop rock",        "north america"),
    ("Kendrick Lamar",          "hip hop",         "north america"),
    ("Drake",                   "hip hop",         "north america"),
    ("SZA",                     "r&b",             "north america"),
    ("Frank Ocean",             "r&b",             "north america"),
    ("Stevie Wonder",           "soul",            "north america"),
    ("Aretha Franklin",         "soul",            "north america"),
    ("Johnny Cash",             "country",         "north america"),
    ("Dolly Parton",            "country",         "north america"),
    ("Miles Davis",             "jazz",            "north america"),
    ("John Coltrane",           "jazz",            "north america"),
    ("Nirvana",                 "rock",            "north america"),
    ("Metallica",               "metal",           "north america"),
    # --- Oceania ----------------------------------------------------------
    ("Tame Impala",             "psych rock",      "oceania"),
    ("Sia",                     "pop",             "oceania"),
    ("AC/DC",                   "rock",            "oceania"),
    ("Kylie Minogue",           "pop",             "oceania"),
    ("Flume",                   "electronic",      "oceania"),
    ("Lorde",                   "art pop",         "oceania"),
    ("Six60",                   "pop",             "oceania"),
    ("Stan Walker",             "r&b",             "oceania"),
]

# Region label for the songs already in data/songs.csv before artist mode
# existed, so merging never leaves the new column blank. Keyed by artist.
# "demo" marks the ten invented placeholder tracks the project started with
# (Neon Echo, LoRoom, ...) -- they are not real releases and have no region.
EXISTING_REGIONS = {
    "neon echo": "demo", "loroom": "demo", "voltline": "demo",
    "paper lanterns": "demo", "max pulse": "demo", "orbit bloom": "demo",
    "slow stereo": "demo", "indigo parade": "demo",
    "taylor swift": "north america", "bruno mars": "north america",
    "the weeknd": "north america", "harry styles": "europe",
    "dua lipa": "europe", "billie eilish": "north america",
    "olivia rodrigo": "north america", "nirvana": "north america",
    "sixpence none the richer": "north america", "drake": "north america",
    "diamond platnumz": "east africa", "sauti sol": "east africa",
    "nyashinski": "east africa", "otile brown": "east africa",
    "burna boy": "west africa", "rema": "west africa",
}
UNKNOWN_REGION = "unknown"

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
DEFAULT_SIZE = 20  # how many search hits to scan per seed when matching artist
DEFAULT_TARGET = 1000    # songs artist mode aims to collect
DEFAULT_PER_ARTIST = 12  # max tracks kept per artist, before round-robin trimming

# CSV columns, in the exact order recommender.load_songs expects.
CSV_FIELDS = [
    "id", "title", "artist", "genre", "region", "mood",
    "energy", "tempo_bpm", "valence", "danceability", "acousticness",
]

RECCOBEATS_API = "https://api.reccobeats.com/v1"
RECCO_BATCH = 40   # ReccoBeats accepts up to 40 ids per audio-features request.
ARTIST_PAGE = 50   # /artist/{id}/track rejects size > 50.

# The audio-features fields the recommender scores on. ReccoBeats sometimes
# returns a features object with some of these set to null (it knows the track
# but not its analysis), so every candidate is checked before it becomes a row.
REQUIRED_FEATURES = ["valence", "energy", "tempo", "danceability", "acousticness"]


def usable_features(feat: Optional[Dict]) -> bool:
    """True when this features object has every number the CSV needs."""
    if not feat:
        return False
    return all(isinstance(feat.get(key), (int, float)) for key in REQUIRED_FEATURES)


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


# --- ReccoBeats: artist discovery -----------------------------------------
def _norm(text: str) -> str:
    """Case-, space- and accent-insensitive key for comparing names.

    Accents are folded so a roster entry typed in plain ASCII ("Rosalia",
    "Beyonce", "Joao Gilberto") still matches ReccoBeats' spelling ("ROSALÍA",
    "Beyoncé", "João Gilberto") -- otherwise most of the non-Anglophone roster
    would be reported as a MISS for punctuation reasons alone.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(folded.lower().split())


def search_artist_ids(name: str, size: int = 10) -> List[str]:
    """Return every ReccoBeats artist id whose name is an exact match for `name`.

    ReccoBeats carries several entries per real artist (one per Spotify id it has
    seen), and usually only one of them holds the actual discography -- the rest
    have a single stray track. So we take all exact-name matches and let the
    caller pool their tracks, rather than trusting the first hit.
    """
    resp = requests.get(
        f"{RECCOBEATS_API}/artist/search",
        params={"searchText": name, "size": size},
        headers={"Accept": "application/json"},
        timeout=20,
    )
    resp.raise_for_status()
    target = _norm(name)
    return [c["id"] for c in resp.json().get("content", []) if _norm(c["name"]) == target]


def _dedupe_key(title: str) -> str:
    """Collapse the many editions of one song ("... - Remastered", "(Live)")."""
    base = title.split(" - ")[0]
    if "(" in base:
        base = base.split("(")[0]
    return _norm(base)


def artist_tracks(name: str, artist_ids: List[str], per_artist: int,
                  pages: int = 2) -> List[Dict]:
    """Pull an artist's tracks and return the most popular `per_artist` of them.

    Only tracks where `name` is the PRIMARY credit are kept: a featured verse on
    someone else's record would otherwise get filed under this artist's genre and
    region, which is exactly the mislabeling this catalog is trying to avoid.
    Duplicate editions of the same song collapse to the most popular one.
    """
    best: Dict[str, Dict] = {}
    for artist_id in artist_ids:
        for page in range(pages):
            resp = requests.get(
                f"{RECCOBEATS_API}/artist/{artist_id}/track",
                params={"size": ARTIST_PAGE, "page": page},
                headers={"Accept": "application/json"},
                timeout=25,
            )
            if not resp.ok:
                break
            content = resp.json().get("content", [])
            for track in content:
                artists = track.get("artists") or []
                if not artists or _norm(artists[0]["name"]) != _norm(name):
                    continue
                key = _dedupe_key(track["trackTitle"])
                current = best.get(key)
                if current is None or track.get("popularity", 0) > current.get("popularity", 0):
                    best[key] = track
            if len(content) < ARTIST_PAGE:
                break  # last page for this id
            time.sleep(0.1)

    ranked = sorted(best.values(), key=lambda t: t.get("popularity", 0), reverse=True)
    return ranked[:per_artist]


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
        if not usable_features(feat):
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
            "region": EXISTING_REGIONS.get(_norm(item["artist"]), UNKNOWN_REGION),
            "mood": derive_mood(valence, energy),
            "energy": round(energy, 3),
            "tempo_bpm": round(float(feat["tempo"]), 1),
            "valence": round(valence, 3),
            "danceability": round(float(feat["danceability"]), 3),
            "acousticness": round(float(feat["acousticness"]), 3),
        })
        next_id += 1
    return rows


def build_from_artists(roster: List[tuple], target: int, per_artist: int) -> List[Dict]:
    """Expand the artist roster into up to `target` real, feature-scored songs.

    Two passes. First we collect a shortlist per artist (their most popular
    tracks) and fetch audio features for the whole pool in one sweep. Then we
    take songs round-robin -- one per artist per lap -- until we reach `target`,
    so the catalog fills out evenly across regions instead of handing the first
    thousand slots to whichever artists happen to sit at the top of the list.
    """
    print(f"Resolving {len(roster)} artists on ReccoBeats (target {target} songs)...")
    shortlists: List[Dict] = []
    for name, genre, region in roster:
        try:
            artist_ids = search_artist_ids(name)
            tracks = artist_tracks(name, artist_ids, per_artist) if artist_ids else []
        except requests.RequestException as exc:
            print(f"  ERR   {name}: {exc}", file=sys.stderr)
            continue
        if not tracks:
            print(f"  MISS  {name} -- no primary-credit tracks in ReccoBeats, skipping",
                  file=sys.stderr)
            continue
        shortlists.append({"name": name, "genre": genre, "region": region,
                           "tracks": tracks})
        print(f"  ok    {name:<26} {len(tracks):>3} tracks  [{region}]")

    if not shortlists:
        return []

    all_ids = [t["id"] for s in shortlists for t in s["tracks"]]
    print(f"Fetching audio features for {len(all_ids)} candidate tracks...")
    features = reccobeats_features(all_ids)

    # Round-robin across artists so every region gets represented. Each artist
    # keeps a cursor into its own shortlist; a track with no audio features just
    # advances that cursor, so an artist never loses its turn to a missing row.
    rows: List[Dict] = []
    seen: set = set()
    cursors = [0] * len(shortlists)
    while len(rows) < target:
        added_this_lap = 0
        for index, shortlist in enumerate(shortlists):
            if len(rows) >= target:
                break
            track = feat = None
            while cursors[index] < len(shortlist["tracks"]):
                candidate = shortlist["tracks"][cursors[index]]
                cursors[index] += 1
                key = (_dedupe_key(candidate["trackTitle"]), _norm(shortlist["name"]))
                if key in seen or not usable_features(features.get(candidate["id"])):
                    continue  # duplicate, or ReccoBeats has no usable analysis
                seen.add(key)
                track, feat = candidate, features[candidate["id"]]
                break
            if track is None:
                continue  # this artist is exhausted
            valence, energy = float(feat["valence"]), float(feat["energy"])
            rows.append({
                "id": 0,  # real ids are assigned by the writer/merger
                "title": track["trackTitle"],
                "artist": track["artists"][0]["name"],
                "genre": shortlist["genre"],
                "region": shortlist["region"],
                "mood": derive_mood(valence, energy),
                "energy": round(energy, 3),
                "tempo_bpm": round(float(feat["tempo"]), 1),
                "valence": round(valence, 3),
                "danceability": round(float(feat["danceability"]), 3),
                "acousticness": round(float(feat["acousticness"]), 3),
            })
            added_this_lap += 1
        if added_this_lap == 0:
            break  # every artist is exhausted

    for index, row in enumerate(rows, start=1):
        row["id"] = index
    print(f"Collected {len(rows)} songs across "
          f"{len({r['region'] for r in rows})} regions.")
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
    for row in existing:  # older CSVs predate the region column
        if not row.get("region"):
            row["region"] = EXISTING_REGIONS.get(_norm(row["artist"]), UNKNOWN_REGION)

    by_key = {(_dedupe_key(r["title"]), _norm(r["artist"])): r for r in existing}
    next_id = max((int(r["id"]) for r in existing), default=0) + 1
    upgraded, added = 0, 0
    for row in fetched:
        key = (_dedupe_key(row["title"]), _norm(row["artist"]))
        match = by_key.get(key)
        if match:
            for col in FEATURE_COLS:
                match[col] = row[col]
            if match.get("region", UNKNOWN_REGION) == UNKNOWN_REGION:
                match["region"] = row["region"]
            upgraded += 1
        else:
            existing.append({**row, "id": next_id})
            by_key[key] = existing[-1]  # so later fetched duplicates fold in too
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
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("artists", "tracks"), default="artists",
                        help="artists: expand ARTIST_ROSTER into --target songs "
                             "(default); tracks: resolve the per-song SEEDS list")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output CSV path (default data/songs.csv)")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="tracks mode: search hits to scan per seed (default 20)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"artists mode: how many songs to collect "
                             f"(default {DEFAULT_TARGET})")
    parser.add_argument("--per-artist", type=int, default=DEFAULT_PER_ARTIST,
                        help=f"artists mode: max songs kept per artist "
                             f"(default {DEFAULT_PER_ARTIST})")
    parser.add_argument("--region", action="append", default=None,
                        help="artists mode: limit to this region (repeatable), "
                             "e.g. --region 'east africa'")
    parser.add_argument("--merge", action="store_true",
                        help="merge into the existing --out CSV (upgrade matches, "
                             "append new, keep ids and unresolved rows) instead of "
                             "overwriting it")
    args = parser.parse_args()

    try:
        if args.mode == "tracks":
            rows = build_catalog(SEEDS, args.size)
        else:
            roster = ARTIST_ROSTER
            if args.region:
                wanted = {r.lower() for r in args.region}
                roster = [a for a in roster if a[2] in wanted]
                if not roster:
                    known = sorted({a[2] for a in ARTIST_ROSTER})
                    print(f"No roster artists in {args.region}. Known regions: "
                          f"{known}", file=sys.stderr)
                    return 1
            rows = build_from_artists(roster, args.target, args.per_artist)
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
