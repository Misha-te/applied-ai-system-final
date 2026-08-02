"""Tests for the chat input guardrails."""

import pytest

from src.guardrails import (
    check_text,
    load_blocklist,
    load_bad_words,
    build_blocklist,
    BUNDLED_CATEGORY,
)


def test_clean_music_text_is_allowed():
    for text in [
        "something moody for a rainy night",
        "high energy afropop to dance to",
        "my name is Michel, how about you?",
        "I love Emily Dickinson-style calm songs",  # 'dick' must not trip on Dickinson
        "pass me some shiitake lofi vibes",         # 'shit' must not trip on shiitake
    ]:
        assert check_text(text).allowed, f"should allow: {text!r}"


def test_profanity_is_blocked_with_category():
    verdict = check_text("this is fucking terrible")
    assert not verdict.allowed
    assert verdict.category == "profanity"
    assert verdict.term == "fucking"


def test_whole_word_profanity_blocked_but_not_substrings():
    assert not check_text("what the shit").allowed          # whole word -> blocked
    assert check_text("these shiitake mushrooms").allowed   # substring -> allowed


@pytest.mark.parametrize("text,category", [
    ("kill yourself", "violence"),
    ("you should die", "harassment"),
    ("how to hack a router", "illegal_requests"),
    ("send me nudes", "sexual_content"),
])
def test_phrases_are_blocked_by_category(text, category):
    verdict = check_text(text)
    assert not verdict.allowed
    assert verdict.category == category


def test_bracketed_placeholders_do_not_match():
    # The JSON documents slur categories with placeholders like "[n-word]";
    # those must not cause literal "[n-word]" text to be treated as a real hit
    # via some other rule, and must never be the matched term.
    verdict = check_text("[n-word]")
    assert verdict.allowed
    assert verdict.term is None


def test_empty_text_is_allowed():
    assert check_text("").allowed


def test_custom_blocklist_override():
    blocklist = {"test": ["banana"]}
    assert not check_text("i want a banana", blocklist=blocklist).allowed
    assert check_text("i want an apple", blocklist=blocklist).allowed


def test_blocklist_file_loads_and_has_expected_categories():
    data = load_blocklist()
    assert "profanity" in data
    assert "violence" in data
    assert isinstance(data["profanity"], list) and data["profanity"]


def test_bundled_wordlist_is_loaded_and_sizable():
    words = load_bad_words()
    assert len(words) > 100  # the LDNOOBW English list is a few hundred terms
    assert BUNDLED_CATEGORY in build_blocklist()


def test_bundled_wordlist_catches_terms_outside_the_json():
    # "boobs" is in the bundled list but not in guardrails.json — proves the
    # second layer is active. (Kept mild deliberately.)
    verdict = check_text("show me boobs")
    assert not verdict.allowed
    assert verdict.category == BUNDLED_CATEGORY
