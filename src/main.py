"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from pathlib import Path

try:  # works as a package (python -m src.main)
    from src.recommender import (
        load_songs, recommend_songs, to_song, build_profile, profile_to_prefs,
    )
except ModuleNotFoundError:  # works as a plain script (python src/main.py)
    from recommender import (
        load_songs, recommend_songs, to_song, build_profile, profile_to_prefs,
    )

try:  # nice bordered tables if tabulate is installed (see requirements.txt)
    from tabulate import tabulate
except ModuleNotFoundError:  # pragma: no cover
    tabulate = None

# Resolve the CSV relative to the project root so it works from any directory.
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

# A few distinct listeners to try the recommender against. Each is a plain
# user-preference dict: the keys line up with the FEATURE_WEIGHTS recipe in
# recommender.py, and any key can be left out.
USER_PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "danceability": 0.85,
        "valence": 0.85,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "danceability": 0.55,
        "acousticness": 0.8,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.9,
        "danceability": 0.6,
        "valence": 0.45,
    },
}


def print_recommendations(name: str, user_prefs: dict, songs: list, k: int = 5) -> None:
    """Print the top-k recommendations for a single named profile as a summary table."""
    header = f"Top Recommendations — {name}"
    print(header)
    print("=" * len(header))
    prefs_summary = ", ".join(f"{key}={value}" for key, value in user_prefs.items())
    print(f"Profile: {prefs_summary}")
    print()

    recommendations = recommend_songs(user_prefs, songs, k=k)
    headers = ["#", "Song", "Artist", "Genre", "Score", "Reasons"]
    rows = []
    for position, (song, score, reasons) in enumerate(recommendations, start=1):
        rows.append([
            position,
            song["title"],
            song["artist"],
            song["genre"],
            f"{score:.2f}%",
            "\n".join(f"- {reason}" for reason in reasons),
        ])

    if tabulate is not None:
        # "grid" keeps every reason on its own line inside the Reasons cell.
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        _print_ascii_table(headers, rows)
    print()


def _print_ascii_table(headers: list, rows: list) -> None:
    """Fallback table for when tabulate is not installed (reasons listed under each row)."""
    top = ["#", "Song", "Artist", "Genre", "Score"]
    widths = [len(h) for h in top]
    for row in rows:
        for i in range(5):
            widths[i] = max(widths[i], len(str(row[i])))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(top))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(5)))
        for reason_line in row[5].splitlines():
            print(f"    {reason_line}")


def demo_learned_profile(songs: list, k: int = 5) -> None:
    """Show the OOP path feeding the functional recommender.

    Instead of a hand-written preference dict, we build a UserProfile from a
    listening history (build_profile), translate it into preferences
    (profile_to_prefs), and hand those to the same recommend_songs used above.
    This is the OOP side and the functional side working as one pipeline.
    """
    song_objects = [to_song(row) for row in songs]
    # A study/chill listener: liked a few lofi tracks, replays "Focus Flow" (9),
    # which counts twice when averaging their taste.
    profile = build_profile(
        name="Learned from listening history (lofi study)",
        songs=song_objects,
        liked_ids=[2, 4, 9],
        replayed_ids=[9],
    )
    prefs = profile_to_prefs(profile)
    print_recommendations(profile.name, prefs, songs, k=k)


def main() -> None:
    songs = load_songs(str(CSV_PATH))

    for name, user_prefs in USER_PROFILES.items():
        print_recommendations(name, user_prefs, songs, k=5)
        print()

    # Same recommender, but driven by a profile learned from listening history.
    demo_learned_profile(songs, k=5)


if __name__ == "__main__":
    main()
