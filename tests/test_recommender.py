"""
Tests for the real recommendation pipeline.

These exercise the functions that actually run when you `python -m src.main`
(load_songs, score_song, recommend_songs) plus the OOP bridge that feeds a
learned profile into them (build_profile -> profile_to_prefs). They do NOT
re-implement any scoring logic; they call the real functions and check outcomes.
"""

from pathlib import Path

import pytest

from src.recommender import (
    Song,
    UserProfile,
    load_songs,
    score_song,
    recommend_songs,
    build_profile,
    profile_to_prefs,
    to_song,
)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


def make_song(**overrides) -> dict:
    """A plain song dict with sensible defaults; override any field per test."""
    song = {
        "id": 1,
        "title": "Song",
        "artist": "Artist",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "tempo_bpm": 120.0,
        "valence": 0.8,
        "danceability": 0.8,
        "acousticness": 0.2,
    }
    song.update(overrides)
    return song


# --- score_song ------------------------------------------------------------

def test_pop_song_scores_higher_than_rock_for_a_pop_fan():
    """A pop-loving profile should score a matching pop song above a rock song."""
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    pop = make_song(id=1, genre="pop", mood="happy", energy=0.8)
    rock = make_song(id=2, genre="rock", mood="intense", energy=0.9)

    pop_score, _ = score_song(prefs, pop)
    rock_score, _ = score_song(prefs, rock)

    assert pop_score > rock_score


def test_score_is_a_percentage_and_reasons_are_generated():
    """Scores stay within 0-100 and every scored feature produces a reason."""
    prefs = {"genre": "pop", "energy": 0.8}
    score, reasons = score_song(prefs, make_song(genre="pop", energy=0.8))

    assert 0.0 <= score <= 100.0
    assert score == 100.0  # perfect match on both scored features
    # One reason per scored feature (genre + energy), and genre match is named.
    assert any("Genre matches your preference for pop" in r for r in reasons)


def test_no_matching_preferences_scores_zero_with_explanation():
    score, reasons = score_song({}, make_song())
    assert score == 0.0
    assert reasons and "No matching preferences" in reasons[0]


# --- recommend_songs: ranking + diversity ---------------------------------

def test_recommend_ranks_by_score_high_to_low():
    prefs = {"genre": "pop", "energy": 0.8}
    songs = [
        make_song(id=1, artist="A", genre="rock", energy=0.2),   # poor match
        make_song(id=2, artist="B", genre="pop", energy=0.8),    # best match
        make_song(id=3, artist="C", genre="pop", energy=0.5),    # middling
    ]
    results = recommend_songs(prefs, songs, k=3, artist_penalty=0.0, genre_penalty=0.0)

    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0]["id"] == 2  # the best-matching song is first


def test_diversity_penalty_reorders_same_artist_songs():
    """With the artist penalty on, a second song by the same artist is demoted
    below a slightly-lower-scoring song by a different artist."""
    prefs = {"energy": 0.80}
    songs = [
        make_song(id=1, artist="Dupe", energy=0.80),   # base 100
        make_song(id=2, artist="Dupe", energy=0.75),   # base ~95
        make_song(id=3, artist="Solo", energy=0.72),   # base ~92
    ]

    # Penalties off: pure score order -> [1, 2, 3].
    plain = recommend_songs(prefs, songs, k=3, artist_penalty=0.0, genre_penalty=0.0)
    assert [s["id"] for s, _, _ in plain] == [1, 2, 3]

    # Artist penalty on: song 2 (same artist as 1) is pushed below song 3.
    diverse = recommend_songs(prefs, songs, k=3, artist_penalty=20.0, genre_penalty=0.0)
    assert [s["id"] for s, _, _ in diverse] == [1, 3, 2]

    # And the demoted song explains why.
    demoted_reasons = diverse[2][2]
    assert any("Diversity penalty" in r for r in demoted_reasons)


def test_recommend_returns_at_most_k():
    prefs = {"genre": "pop"}
    songs = [make_song(id=i, artist=f"A{i}") for i in range(5)]
    assert len(recommend_songs(prefs, songs, k=2)) == 2


@pytest.mark.parametrize("bad_k", [True, 1.5, "3"])
def test_recommend_rejects_non_integer_k(bad_k):
    with pytest.raises(TypeError):
        recommend_songs({}, [make_song()], k=bad_k)


def test_recommend_rejects_negative_k():
    with pytest.raises(ValueError):
        recommend_songs({}, [make_song()], k=-1)


# --- load_songs ------------------------------------------------------------

def test_load_songs_parses_numeric_fields():
    songs = load_songs(str(CSV_PATH))
    assert len(songs) > 0
    first = songs[0]
    assert isinstance(first["id"], int)
    assert isinstance(first["energy"], float)


# --- OOP bridge: build_profile -> profile_to_prefs -> recommend_songs ------

def test_build_profile_averages_taste_and_weights_replays():
    songs = [
        Song(1, "Calm", "X", "lofi", "chill", 0.2, 80, 0.5, 0.5, 0.9),
        Song(2, "Loud", "Y", "rock", "intense", 0.8, 140, 0.6, 0.6, 0.1),
    ]
    # Song 2 replayed -> counts twice, pulling the average toward high energy.
    profile = build_profile("Rae", songs, liked_ids=[1, 2], replayed_ids=[2])
    # weighted energy = (0.2 + 0.8 + 0.8) / 3 = 0.6
    assert profile.avg_energy == pytest.approx(0.6, abs=1e-3)


def test_profile_flows_into_recommender():
    """The OOP path produces recommendations from the same functional core."""
    catalog = [to_song(row) for row in load_songs(str(CSV_PATH))]
    profile = build_profile("Study", catalog, liked_ids=[2, 4, 9], replayed_ids=[9])
    prefs = profile_to_prefs(profile)

    song_dicts = load_songs(str(CSV_PATH))
    results = recommend_songs(prefs, song_dicts, k=3)

    assert len(results) == 3
    # Every result carries a score and at least one human-readable reason.
    for song, score, reasons in results:
        assert 0.0 <= score <= 100.0
        assert reasons
