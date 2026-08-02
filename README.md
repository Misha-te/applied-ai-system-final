# 🎵 MusiReco — Music Recommender Simulation

*A content-based music recommender that ranks songs to a listener's taste, learns
its own scoring weights from data, and explains every pick in plain English —
runnable as a command-line tool or a conversational "DJ" web app.*

---

## The Original Project (Modules 1–3)

This project began as the **Music Recommender Simulation ("MusiReco 1.0")** for
Modules 1–3. Its original goal was to represent songs and a listener's *taste
profile* as data, then design a **scoring rule** that turns that data into ranked
recommendations — each with a 0–100% match score and a short, human-readable
reason. That first version was a **command-line, content-based recommender** with
hand-picked feature weights, built to explore how real recommender systems (Spotify,
TikTok) turn preferences into predictions.

This final extends that foundation into a fuller **applied AI system**: the
hand-picked weights are now **learned by a trained model**, the recommender has an
interactive **LLM-powered web front-end** with **safety guardrails**, and the whole
thing is documented with a system diagram, tests, and a model card.

---

## Title and Summary — What It Does and Why It Matters

**MusiReco** takes a listener's taste — favorite genre and mood, plus how energetic,
positive, danceable, and acoustic they like their music — and returns the **top *k*
songs** from its catalog, each with a **match percentage** and a list of reasons for
that score.

Why it matters: recommenders are one of the most widely deployed forms of AI, and
they're often opaque ("why am I being shown this?"). MusiReco is a small, fully
**transparent** version of that idea — every score is explainable, the model that
sets the weights is a few dozen lines of readable code, and the ways it can be
biased are documented rather than hidden. It's built for **classroom exploration**,
not production use.

---

## Architecture Overview

The full diagram lives in [diagrams/system-architecture.md](diagrams/system-architecture.md)
(Mermaid source: [system-architecture.mmd](diagrams/system-architecture.mmd)); the
scoring math is sketched in [sketch design.mmd](diagrams/sketch%20design.mmd).

Data flows **input → process → output**, with automated tests and a human checking
the AI's results along the way:

- **Input — listener taste.** Either CLI preset profiles ([src/main.py](src/main.py))
  or answers typed into the web DJ chat ([src/app.py](src/app.py)).
- **Agent (web only).** The conversational DJ collects taste. Free-text answers pass
  through **guardrails** ([src/guardrails.py](src/guardrails.py)) before reaching an
  optional **LLM** (Gemini or DeepSeek) that turns natural language into preferences.
- **Retriever.** `load_songs()` reads the song catalog ([data/songs.csv](data/songs.csv))
  that supplies the candidate songs.
- **Specialized model (trained offline).** [src/train_weights.py](src/train_weights.py)
  trains a logistic-regression model on labeled taste data and writes the learned
  feature weights to [data/learned_weights.json](data/learned_weights.json).
- **Recommender core.** [src/recommender.py](src/recommender.py) scores every song
  (`score_song`) using those learned weights and ranks the top *k* with a diversity
  penalty (`recommend_songs`).
- **Output.** The top-*k* songs, each with a score % and plain-English reasons.
- **Testing & human evaluation.** `pytest` checks the core and guardrails; the
  trainer reports accuracy; a human reads the reasons to judge whether picks *feel*
  right, and that judgment feeds back into the training labels.

### How the scoring works (the recipe)

For each song, MusiReco scores **each feature from 0 to 1**, then blends them with a
weight per feature into a single 0–100% score:

- **Categorical (genre, mood):** `1.0` if it matches the listener's preference, else `0.0`.
- **Numeric (energy, valence, danceability, acousticness):** `1 − |song − target|` on
  the shared 0–1 scale — "how close is it."

The weights (how much each feature counts) are the part that's **learned**, not
guessed — see [Design Decisions](#design-decisions).

---

## Setup Instructions

### 1. Install

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# install dependencies
pip install -r requirements.txt
```

### 2. Run the command-line demo

```bash
python -m src.main
```

Scores the catalog for three sample listeners and one profile *learned from
listening history*, printing ranked tables with reasons.

### 3. (Optional) Retrain the scoring model

```bash
python -m src.train_weights
```

Retrains the weights and refreshes `data/learned_weights.json`. The recommender
picks it up automatically on the next run; if the file is missing, it falls back to
the original hand-tuned weights so everything still works out of the box.

### 4. Run the web app (conversational DJ)

```bash
streamlit run src/app.py
```

A sidebar picks the provider:

| Provider | Needs a key? | What it adds |
|----------|:------------:|--------------|
| **Free (no AI)** | No | Preset buttons + scripted questions. Works out of the box. |
| **Gemini** | `GEMINI_API_KEY` | Reads free-text answers and personalizes wording. |
| **DeepSeek** | `DEEPSEEK_API_KEY` | Same, via the OpenAI-compatible API. |

Keys go in `.streamlit/secrets.toml` (gitignored — never commit them):

```toml
GEMINI_API_KEY   = "your-gemini-key"     # https://aistudio.google.com/app/apikey
DEEPSEEK_API_KEY = "your-deepseek-key"   # https://platform.deepseek.com/api_keys
```

The ranked list is identical across providers when you only tap presets — the
provider changes the *wording* and *free-text understanding*, never the scoring.

### 5. Run the tests

```bash
pytest
```

---

## Sample Interactions

### Example 1 — CLI, "High-Energy Pop" listener

**Input:** `genre=pop, mood=happy, energy=0.9, danceability=0.85, valence=0.85`

**Output (top 3 of 5):**

```text
1  Sunrise City    Neon Echo   pop        95.67%
    - Genre matches your preference for pop
    - Energy is very close to your preferred level
    - Mood matches your preference for happy
2  Levitating      Dua Lipa    pop        85.10%
    - Genre matches your preference for pop
    - Diversity penalty: -10 because its genre pop already appears above
3  Calm Down       Rema        afrobeats  74.80%
    - Genre (afrobeats) differs from your preferred pop
    - Valence is very close to your preferred level
    - Mood matches your preference for happy
```

Note the **diversity penalty**: a second pop song is docked so the list doesn't pile
up on one genre, letting an afrobeats track with the right *feel* surface.

### Example 2 — CLI, "Chill Lofi" listener

**Input:** `genre=lofi, mood=chill, energy=0.35, danceability=0.55, acousticness=0.8`

**Output (top 2 of 5):**

```text
1  Library Rain     Paper Lanterns  lofi     98.26%
    - Genre matches your preference for lofi
    - Energy is very close to your preferred level
    - Acousticness is very close to your preferred level
2  Midnight Coding  LoRoom          lofi     84.57%
    - Genre matches your preference for lofi
    - Diversity penalty: -10 because its genre lofi already appears above
```

The near-opposite taste to Example 1 produces a completely different list — a good
sign the scoring actually responds to the profile.

### Example 3 — Training the scoring model

**Input:** `python -m src.train_weights`

**Output:**

```text
Trained on 128 (listener, song) examples from 4 listeners (24 likes, 104 non-likes).
Train accuracy: 89.1%

feature         hand-tuned   learned
------------------------------------
genre                 0.30      0.17
energy                0.20      0.29
valence               0.15      0.10
danceability          0.15      0.19
mood                  0.15      0.08
acousticness          0.05      0.17
```

The model *disagreed* with the original human guesses — energy and acousticness
matter more than assumed, genre and mood less.

### Example 4 — Web DJ (Free mode), a typical guided session

```text
DJ:   Hey! I'm your DJ. What should I call you?
You:  Amina
DJ:   Nice to meet you, Amina. What mood are you in?      [happy] [chill] [intense] …
You:  (taps "happy")
DJ:   How much energy do you want?                        [low] [medium] [high]
You:  (taps "high")
…
DJ:   Alright Amina — here are your picks:
      1. Sunrise City — 95.67% (genre + energy + happy mood all match)
      2. Levitating   — 85.10% …
```

With an LLM provider on, you can instead type *"something upbeat for a morning run"*
and the DJ maps it to the same preference fields.

---

## Design Decisions

- **Percentages instead of points.** The starter used a points total (+2 genre, +1
  mood…), but `3.4` means nothing on its own. A **0–100% score** reads directly as
  "how likely you are to like this," which is both more honest and easier to explain.
- **Content-based, not collaborative.** With no real user base, I score songs by
  their *features* against a taste profile. *Trade-off:* it can't do "people like you
  also liked…" — I note collaborative filtering as future work.
- **Learned weights over hand-tuned ones.** The biggest design change: instead of me
  guessing that genre is worth 30%, a **logistic-regression model learns the weights
  from labeled taste data**. *Trade-off:* the dataset is tiny (4 listeners, 32 songs,
  labels I curated), so accuracy is in-sample — it proves the *method*, not
  production-grade generalization. I kept the hand-tuned weights as a **fallback** so
  the app never breaks if the model file is absent.
- **Pure-Python model, no ML library.** The trainer is a few dozen lines of gradient
  descent — no scikit-learn. *Trade-off:* slower and more manual, but fully
  transparent and dependency-light, which matters more for a learning project.
- **Diversity penalty.** Ranking docks repeat artists/genres so the top-*k* isn't five
  near-identical songs. *Trade-off:* the single best match can be pushed down a slot.
- **Guardrails before the LLM.** Because the chat sends free text to a model, input is
  screened against a blocklist + the LDNOOBW wordlist first. *Trade-off:* keyword
  filtering can over- or under-block, so it uses word boundaries to avoid false hits
  like "Dickinson."
- **Provider-agnostic AI.** Gemini and DeepSeek plug into the *same* scoring engine;
  the LLM only handles wording and free-text understanding, never the ranking. This
  keeps results reproducible and the AI swappable.

---

## Testing Summary

**Automated tests** ([tests/](tests/)) run against the *real* code paths, not stubs:

- `test_recommender.py` — `score_song`, `recommend_songs` (ranking **and** the
  diversity penalty), `load_songs`, and the `build_profile → profile_to_prefs` bridge.
- `test_guardrails.py` — clean text passes, each blocklist category is caught, the
  word-boundary false-positive guard holds, and the bundled wordlist layer is active.

**What worked:** all 26 tests pass, and they're robust to the learned weights — I
verified the suite stays green whether the model file is present or the fallback is
used. The trainer reaching **89% in-sample accuracy** confirmed the learning pipeline
works end to end.

**What didn't (and what I learned):**

- **The recommender is surprisingly insensitive to weight tuning.** In an early
  experiment I doubled energy and halved genre, and the *rankings barely moved* —
  the catalog is small and clustered, so the top picks agree on everything at once.
  Lesson: a small dataset can make a system *look* like it's learning when it isn't.
- **Learning challenged my assumptions.** I was confident genre should dominate; the
  data said energy and acousticness matter more. Lesson: even a toy trained model can
  correct a human's priors, and being able to *defend the numbers with data* beats
  defending a guess.
- **Contradictory / out-of-range input isn't rejected.** Profiles like `energy=2.0`
  or "high energy + sad mood" still produce a confident score, because features are
  scored independently. Lesson: input validation is its own layer of work.

---

## Reflection

Building this made concrete how much of a recommendation is just **probability plus
ranking**, and how quickly bias creeps in through the data and the weights rather than
the code. The most valuable step was replacing my hand-picked weights with a trained
model — watching the data disagree with me was the moment it stopped feeling like
magic.

> 📌 My full **responsible-AI reflection** — how I collaborated with AI, one helpful
> and one flawed AI suggestion, and the system's limitations — is in the
> [**model card**](model_card.md), which also holds the detailed evaluation.

---

## Credits

- Profanity/slur coverage uses the **LDNOOBW** wordlist (*"List of Dirty, Naughty,
  Obscene, and Otherwise Bad Words"*,
  <https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words>),
  bundled at [data/bad_words.txt](data/bad_words.txt).
- Optional AI providers: **Google Gemini** (`google-genai`) and **DeepSeek** (via the
  OpenAI-compatible `openai` SDK).
