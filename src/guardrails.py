"""
Input guardrails for the Music DJ chat.

The chat lets users type free text, which is then sent to an LLM and used to
build recommendations. This module screens that text before acting on it, using
two layers:
  1. data/guardrails.json  — hand-curated categories (profanity, slurs, sexual/
     violent content, harassment, illegal requests).
  2. data/bad_words.txt     — the bundled open-source LDNOOBW wordlist, for
     broad slur/profanity coverage without hand-typing slurs into the repo.

Design notes:
- Single words are matched on WORD BOUNDARIES, so "dick" does not flag
  "Dickinson" and "shit" does not flag "shiitake".
- Multi-word phrases ("kill yourself") are matched as substrings on the
  whitespace-normalized text.
- Bracketed placeholders like "[n-word]" in the JSON are skipped, so the file
  can document a category without this file containing the actual slur. Replace
  them with real terms to enable matching for those categories.

This is a deliberately simple, transparent filter — it does not catch creative
obfuscation (leetspeak, spacing tricks). It is a first line of defense, not a
complete moderation system.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BLOCKLIST_PATH = DATA_DIR / "guardrails.json"
# Bundled open-source wordlist (LDNOOBW "List of Dirty, Naughty, Obscene, and
# Otherwise Bad Words", English). Broadens slur/profanity coverage beyond the
# hand-curated categories above. Optional — the module works without it.
BAD_WORDS_PATH = DATA_DIR / "bad_words.txt"
BUNDLED_CATEGORY = "bundled_wordlist"


@dataclass
class Verdict:
    """Result of screening one piece of text."""
    allowed: bool
    category: Optional[str] = None   # e.g. "profanity" when blocked
    term: Optional[str] = None       # the blocklist entry that matched


def load_blocklist(path: Path = BLOCKLIST_PATH) -> Dict[str, List[str]]:
    """Load the categorized blocklist from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_bad_words(path: Path = BAD_WORDS_PATH) -> List[str]:
    """Load the bundled flat wordlist (one term per line). Returns [] if the
    file isn't present, so the bundled layer is purely optional."""
    if not path.exists():
        return []
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            words.append(term)
    return words


def build_blocklist() -> Dict[str, List[str]]:
    """Combine the curated JSON categories with the bundled wordlist (if any)."""
    blocklist = load_blocklist()
    bundled = load_bad_words()
    if bundled:
        blocklist = {**blocklist, BUNDLED_CATEGORY: bundled}
    return blocklist


# Loaded once at import; pass an explicit blocklist to check_text() to override.
_BLOCKLIST = build_blocklist()


def _normalize(text: str) -> str:
    """Lowercase and collapse runs of whitespace to single spaces."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_placeholder(term: str) -> bool:
    """Bracketed entries like "[n-word]" document a category without the slur."""
    return term.startswith("[") and term.endswith("]")


def check_text(text: str, blocklist: Optional[Dict[str, List[str]]] = None) -> Verdict:
    """Screen text against the blocklist. Returns a Verdict; when blocked it
    names the category and the specific term that matched."""
    if blocklist is None:
        blocklist = _BLOCKLIST
    if not text:
        return Verdict(allowed=True)

    normalized = _normalize(text)
    for category, terms in blocklist.items():
        for term in terms:
            cleaned = _normalize(term)
            if not cleaned or _is_placeholder(cleaned):
                continue
            if " " in cleaned:  # phrase: substring match
                if cleaned in normalized:
                    return Verdict(False, category, term)
            else:  # single word: whole-word match only
                if re.search(rf"\b{re.escape(cleaned)}\b", normalized):
                    return Verdict(False, category, term)
    return Verdict(allowed=True)
