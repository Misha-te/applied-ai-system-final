"""
Integrity checks on the shipped catalog, data/songs.csv.

The catalog is no longer something you can eyeball -- it is over a thousand rows
pulled from the ReccoBeats API by src/fetch_songs.py. These tests are the
substitute for reading it: they fail if a re-fetch ever writes a malformed row,
renumbers the ids the trainer's labels point at, or quietly narrows the catalog
back down to one part of the world.
"""

from pathlib import Path

import pytest

from src.recommender import load_songs

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

# Ids 1-32 are the original hand-built catalog. src/train_weights.py labels its
# training listeners by these ids, and recommender.sample_profiles cites them as
# replayed songs, so a re-fetch must never renumber them.
ORIGINAL_ID_COUNT = 32

UNIT_FIELDS = ["energy", "valence", "danceability", "acousticness"]


@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(CSV_PATH))


def test_catalog_is_large_and_global(catalog):
    """A re-fetch that collapses the catalog back to a handful of rows fails here."""
    assert len(catalog) >= 500
    regions = {song.get("region", "") for song in catalog}
    assert len(regions - {"", "demo", "unknown"}) >= 8


def test_every_row_is_complete(catalog):
    """No blank cells: every song has all its descriptive and numeric fields."""
    for song in catalog:
        for field in ("title", "artist", "genre", "region", "mood"):
            assert str(song.get(field, "")).strip(), f"{field} missing on id {song['id']}"


def test_numeric_features_are_in_range(catalog):
    """Audio features stay on the 0-1 scale score_song assumes, tempo stays positive."""
    for song in catalog:
        for field in UNIT_FIELDS:
            value = song[field]
            assert 0.0 <= value <= 1.0, f"{field}={value} out of range on id {song['id']}"
        assert song["tempo_bpm"] > 0, f"bad tempo on id {song['id']}"


def test_ids_are_unique_and_sequential(catalog):
    ids = [song["id"] for song in catalog]
    assert len(set(ids)) == len(ids)
    assert ids == list(range(1, len(ids) + 1))


def test_original_training_ids_are_preserved(catalog):
    """The first 32 rows keep their ids so the trainer's labels still point at them."""
    assert len(catalog) >= ORIGINAL_ID_COUNT
    by_id = {song["id"]: song for song in catalog}
    assert by_id[9]["title"] == "Focus Flow"      # Chris's replayed song
    assert by_id[12]["artist"] == "Taylor Swift"  # Amina's replayed song
    assert by_id[17]["artist"] == "Sauti Sol"     # Baraka's replayed song


def test_no_duplicate_songs(catalog):
    """The same title by the same artist should appear once."""
    seen = {(song["title"].lower(), song["artist"].lower()) for song in catalog}
    assert len(seen) == len(catalog)
