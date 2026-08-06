# 🎵 MuxiReco — Music Recommender Simulation

*A content-based music recommender that ranks songs to a listener's taste, learns
its own scoring weights from data, and explains every pick in plain English —
runnable as a command-line tool or a conversational "DJ" web app.*

**🚀 Live demo (Streamlit Community Cloud):** <https://share.streamlit.io/user/misha-te>

---

## The Original Project (Modules 1–3)

This project began as the **Music Recommender Simulation ("MuxiReco 1.0")** for
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

**MuxiReco** takes a listener's taste — favorite genre and mood, plus how energetic,
positive, danceable, and acoustic they like their music — and returns the **top *k*
songs** from its catalog, each with a **match percentage** and a list of reasons for
that score.

Why it matters: recommenders are one of the most widely deployed forms of AI, and
they're often opaque ("why am I being shown this?"). MuxiReco is a small, fully
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
  that supplies the candidate songs — **1,030 real tracks by 193 artists across 13
  world regions**, built by [src/fetch_songs.py](src/fetch_songs.py) from the
  ReccoBeats API (see [Data](#data)).
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

For each song, MuxiReco scores **each feature from 0 to 1**, then blends them with a
weight per feature into a single 0–100% score:

- **Categorical (genre, mood):** `1.0` if it matches the listener's preference, else `0.0`.
- **Numeric (energy, valence, danceability, acousticness):** `1 − |song − target|` on
  the shared 0–1 scale — "how close is it."

The weights (how much each feature counts) are the part that's **learned**, not
guessed — see [Design Decisions](#design-decisions).

---

## Data

The catalog is **1,030 songs by 193 artists**, and every audio number in it is real:
[src/fetch_songs.py](src/fetch_songs.py) pulls titles and Spotify-style audio features
(energy, valence, danceability, acousticness, tempo) from the **ReccoBeats API**, which
needs no API key. Spotify itself closed its audio-features endpoints to new apps in
Nov 2024, which is why ReccoBeats stands in.

Two fields are ours, because no API supplies them cleanly: **genre** and **region**
come from a hand-curated 189-artist roster in `fetch_songs.py`, and **mood** is derived
from valence and energy by `derive_mood()`.

**Regional spread** — the point of the roster is that the catalog doesn't sound like one
radio market:

| region | songs | region | songs |
|---|---:|---|---:|
| west africa | 157 | south asia | 65 |
| east africa | 138 | east asia | 65 |
| latin america | 121 | caribbean | 57 |
| north america | 96 | southeast asia | 45 |
| north africa & middle east | 72 | oceania | 40 |
| europe | 68 | central africa | 30 |
| southern africa | 66 | *demo (invented seed tracks)* | 10 |

Rebuild or extend it:

```bash
python src/fetch_songs.py --merge                      # top up all regions
python src/fetch_songs.py --region "east africa" --target 100 --merge
```

`--merge` keeps existing ids (the trainer's labels point at ids 1–32), upgrades matched
rows in place, and appends the rest. An artist is only used when ReccoBeats returns an
exact name match **and** the artist is the primary credit on the track, so a featured
verse never gets filed under the wrong region; artists that can't be resolved are
logged as `MISS` and skipped, never guessed.

**Honest gaps.** The 10 original placeholder tracks (Neon Echo, LoRoom…) are invented,
not real releases — they carry the region `demo` and are excluded from the region
counts above. Six roster artists don't resolve in ReccoBeats at all (Beyoncé, Björk,
Sigur Rós, Utada Hikaru, Sơn Tùng M-TP, Hoàng Thùy Linh), and `IU` is rejected by the
API's minimum query length, so those artists are simply absent. Region is a label on
the *artist*, not a claim about a song's musical origin.

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

Already deployed — no install needed: <https://share.streamlit.io/user/misha-te>

To run it locally instead:

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
1  Sunrise City  Neon Echo         pop        demo         95.60%
    - Genre matches your preference for pop
    - Energy is very close to your preferred level
    - Mood matches your preference for happy
2  Levitating    Dua Lipa          pop        europe       84.82%
    - Genre matches your preference for pop
    - Diversity penalty: -10 because its genre pop already appears above
3  Nkwatako      Sheebah Karungi   afropop    east africa  78.43%
    - Genre (afropop) differs from your preferred pop
    - Energy is very close to your preferred level
    - Mood matches your preference for happy
```

Note the **diversity penalty**: a second pop song is docked so the list doesn't pile
up on one genre. With a 1,030-song catalog spanning 13 regions, that pushes the list
past the pop shelf into afropop and bongo picks that still fit the numbers — which is
what the **region** label is there to make visible.

### Example 2 — CLI, "Chill Lofi" listener

**Input:** `genre=lofi, mood=chill, energy=0.35, danceability=0.55, acousticness=0.8`

**Output (top 3 of 5):**

```text
1  Library Rain     Paper Lanterns  lofi   demo           98.08%
    - Genre matches your preference for lofi
    - Energy is very close to your preferred level
    - Acousticness is very close to your preferred level
2  Midnight Coding  LoRoom          lofi   demo           84.28%
    - Genre matches your preference for lofi
    - Diversity penalty: -10 because its genre lofi already appears above
3  Ivy              Frank Ocean     r&b    north america  79.25%
    - Genre (r&b) differs from your preferred lofi
    - Acousticness is very close to your preferred level
```

The near-opposite taste to Example 1 produces a completely different list — a good
sign the scoring actually responds to the profile.

### Example 3 — Training the scoring model

**Input:** `python -m src.train_weights`

**Output:**

```text
Trained on 4120 (listener, song) examples from 4 listeners (24 likes, 4096 non-likes).
Balanced accuracy: 86.8%   (likes and non-likes weighted equally)
Plain accuracy:    81.9%   (always-say-no baseline: 99.4%)

feature         hand-tuned   learned
------------------------------------
genre                 0.30      0.17
energy                0.20      0.29
valence               0.15      0.12
danceability          0.15      0.19
mood                  0.15      0.05
acousticness          0.05      0.19
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
  from labeled taste data**. *Trade-off:* the labels are tiny (4 listeners, 24 liked
  songs), so accuracy is in-sample — it proves the *method*, not production-grade
  generalization. I kept the hand-tuned weights as a **fallback** so the app never
  breaks if the model file is absent.
- **Balanced class weights in the trainer.** Growing the catalog to 1,030 songs made
  the training set 99.4% negatives, and the unweighted model took the shortcut of
  predicting "no" for everything — 98% accuracy with the valence and danceability
  weights collapsed to zero. Weighting each class by `n / (2 · class_count)` fixed it,
  and the headline metric is now **balanced accuracy** (an always-say-no model scores
  0.50 there, not 0.98). *Trade-off:* balanced accuracy looks lower than the old
  number; it is simply the honest one.
- **A real API instead of hand-typed rows.** Every audio feature comes from ReccoBeats
  rather than my estimate of how energetic a song feels. *Trade-off:* the catalog is
  limited to what that API carries — a handful of roster artists (Beyoncé, Björk,
  Sơn Tùng M-TP…) can't be resolved at all and are missing rather than approximated.
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
- `test_ai_reliability.py` — the AI-specific safety and reliability layer:
  LLM-output validation (`_clean_prefs` clamps out-of-range numbers, drops unknown
  genres/moods, and never crashes on garbage), **confidence scoring**
  (`confidence_label` thresholds), the trained model (weights are a valid
  non-negative recipe summing to 1; retraining clears a minimum accuracy and is
  deterministic), and end-to-end **reliability** via `src/evaluate.py`.

- `test_catalog.py` — integrity of the 1,030-row catalog, which is far too big to
  eyeball: no blank cells, features on the 0–1 scale, unique sequential ids, ids 1–32
  unchanged so the trainer's labels still point at the right songs, and at least 8
  regions represented.

**How reliability is measured.** `python -m src.evaluate` ranks the catalog for each
of the four labeled taste profiles and checks where their known-liked songs land
(precision/recall@5), plus the recommender's own top-1 confidence. It reports **two
pools**, because precision is a closed-world metric and the like-lists were curated
when the catalog was 32 songs:

- **labeled pool** (ids 1–32) — the slice the labels actually cover. This is the
  reliability guard the test asserts on.
- **full catalog** (1,030 songs) — context only. It is *lower by construction*: a
  perfect unlabeled pick counts as a miss.

**One-line summary of results:**

> **47 of 47 tests pass.** The trained model reaches **86.8% balanced accuracy**;
> across the four labeled listeners the recommender recovers known taste at
> **mean precision@5 = 0.75, recall@5 = 0.67** on the labeled pool, with every top
> pick rated **"High" confidence** (mean top score 96.0%). Against the full catalog
> the same profiles score 0.35 — not a regression but a **labeling gap**: the
> afropop listener's full-catalog top-5 is now Miriam Makeba, Eddy Kenzo, Willy Paul
> and Otile Brown, all genuinely on-taste and none of them on a 32-song like-list.
> Validation rules were what stopped out-of-range model output from reaching the
> scorer.

**What didn't (and what I learned):**

- **The recommender used to be insensitive to weight tuning — because the catalog was
  too small.** In an early experiment I doubled energy and halved genre and the
  *rankings barely moved*: with 32 clustered songs the top picks agreed on everything
  at once. At 1,030 songs that stopped being true; reweighting now visibly reshuffles
  the list. Lesson: a small dataset can make a system *look* like it's learning when
  it isn't, and the fix was more data, not more tuning.
- **Adding data broke a metric, and that was informative.** Precision@5 fell from 0.80
  to 0.35 the moment the catalog grew, because the like-lists still described only 32
  songs. Nothing about the recommender got worse — the evaluation had silently become
  a test of label coverage. Lesson: when you scale the data, re-read what your metric
  is actually measuring.
- **Learning challenged my assumptions.** I was confident genre should dominate; the
  data said energy and acousticness matter more. Lesson: even a toy trained model can
  correct a human's priors, and being able to *defend the numbers with data* beats
  defending a guess.
- **Contradictory / out-of-range input needed its own layer.** Free-text answers go
  through an LLM, which can return nonsense like `energy=2.5` or an invented genre.
  I added `_clean_prefs` to clamp and drop that before it reaches the scorer, and
  `test_ai_reliability.py` now proves it. The *direct numeric* profile path still
  scores contradictory tastes (e.g. "high energy + sad mood") independently without
  complaint — a known limitation. Lesson: input validation is its own layer of work,
  and it belongs right where untrusted data (the model's output) enters the system.

---

## Reproducible Execution Evidence

So the system can be graded **without watching a demo video**, this section shows the
exact commands, inputs, and outputs. Every block below was captured by actually running
the code; the full, unedited logs live in [logs/](logs/) and can be regenerated with the
commands shown.

### Commands to reproduce everything

```bash
pip install -r requirements.txt         # one-time setup

python -m src.main            # CLI demo — ranks 3 preset listeners   -> logs/cli_demo.txt
python -m src.train_weights   # (re)train the scoring weights         -> logs/train_weights.txt
python -m src.evaluate        # reliability: precision/recall @ top-5 -> logs/evaluate.txt
pytest -v                     # 47 automated tests                    -> logs/tests.txt

python src/fetch_songs.py --merge   # rebuild/extend the catalog from ReccoBeats
```

### 1. CLI recommendations — example input → output

**Input** (a preset listener profile from [src/main.py](src/main.py)):

```python
"High-Energy Pop": {"genre": "pop", "mood": "happy",
                    "energy": 0.9, "danceability": 0.85, "valence": 0.85}
```

**Output** (`python -m src.main`, first profile; full run in [logs/cli_demo.txt](logs/cli_demo.txt)):

```text
Top Recommendations — High-Energy Pop
=====================================
Profile: genre=pop, mood=happy, energy=0.9, danceability=0.85, valence=0.85

  # | Song         | Artist          | Genre     | Region      | Score   | Reasons
----+--------------+-----------------+-----------+-------------+---------+---------------------------
  1 | Sunrise City | Neon Echo       | pop       | demo        | 95.60%  | Genre matches pop; energy/
    |              |                 |           |             |         | valence very close; happy
  2 | Levitating   | Dua Lipa        | pop       | europe      | 84.82%  | strong pop match; -10
    |              |                 |           |             |         | diversity penalty (pop above)
  3 | Nkwatako     | Sheebah Karungi | afropop   | east africa | 78.43%  | genre differs; energy + mood fit
  4 | Oh My Gawd   | Mr Eazi         | afrobeats | west africa | 77.79%  | genre differs; numerics fit
  5 | Waah!        | Diamond Platnumz| bongo     | east africa | 77.43%  | genre differs; mood happy
```

Every pick carries a **score %** and **plain-English reasons** — including *why* a
same-genre song was demoted by the diversity penalty.

### 2. Trained model — learned weights (`python -m src.train_weights`)

```text
Trained on 4120 (listener, song) examples from 4 listeners (24 likes, 4096 non-likes).
Balanced accuracy: 86.8%   (likes and non-likes weighted equally)
Plain accuracy:    81.9%   (always-say-no baseline: 99.4%)

feature         hand-tuned   learned
------------------------------------
genre                 0.30      0.17
energy                0.20      0.29
valence               0.15      0.12
danceability          0.15      0.19
mood                  0.15      0.05
acousticness          0.05      0.19
```

The data disagreed with my priors: **energy and acousticness** matter more than I
assumed, **genre and mood** less. Full log: [logs/train_weights.txt](logs/train_weights.txt).

### 3. Reliability results (`python -m src.evaluate`)

Ranks the catalog for each labeled listener and checks where their **known-liked**
songs land (full log: [logs/evaluate.txt](logs/evaluate.txt)):

```text
Recommender reliability (labeled pool) — top-5, 4 labeled listeners (32 songs)

listener               likes  hits   prec  recall   top%  conf
--------------------------------------------------------------
Upbeat pop                 6     3   0.60    0.50   97.3  High
Chill study lofi           5     5   1.00    1.00   99.2  High
Intense rock               4     3   0.60    0.75   97.3  High
Afrobeats / afropop        9     4   0.80    0.44   90.1  High
--------------------------------------------------------------
MEAN                                 0.75    0.67   96.0

Same profiles, ranked against the FULL catalog (1030 songs)
MEAN                                 0.35    0.34   97.6
```

**Read: mean precision@5 = 0.75, recall@5 = 0.67 on the labeled pool.** The Afrobeats
recall (0.44) is a measurement artifact — that listener has 9 likes, more than a top-5
list can hold — not a model failure. (In-sample check: the profiles that trained the
weights.)

**The 0.35 full-catalog number is a labeling gap, and worth stating plainly.** Growing
the catalog from 32 to 1,030 songs added hundreds of tracks that fit these listeners
but were never on anyone's like-list, so precision measured against those lists *must*
fall. Spot-checking confirms it: the afropop listener's full-catalog top-5 is Miriam
Makeba, Eddy Kenzo, Willy Paul, Otile Brown and Willy Paul again — five genuine
afropop tracks scoring 0.00. The honest fix is more labels, not fewer songs.

### 4. Guardrail & validation results

Two safety layers, captured in [logs/guardrails_demo.txt](logs/guardrails_demo.txt):

```text
=== GUARDRAIL: free-text screening (src/guardrails.py) ===
  [clean request]     'I want happy upbeat pop for a road trip' -> ALLOWED
  [profanity]         'this is fucking great music'  -> BLOCKED (category=profanity, term='fucking')
  [violence]          'kill yourself'                -> BLOCKED (category=violence, term='kill yourself')
  [false-positive]    'songs about Dickinson poems'  -> ALLOWED   (word-boundary guard: 'dick' inside a word)
  [false-positive]    'shiitake mushroom jazz'       -> ALLOWED   (word-boundary guard: 'shit' inside a word)

=== LLM-OUTPUT VALIDATION: _clean_prefs clamps/drops unsafe model output ===
  {'genre':'pop','energy':2.5,'valence':-1.0}  -> {'genre':'pop','energy':1.0,'valence':0.0}  (clamped 0..1)
  {'genre':'polka','mood':'grumpy'}            -> {}   (genre/mood not in catalog -> dropped)
  'not even a dict'                            -> {}   (garbage type -> no crash)
```

### 5. Automated test results (`pytest -v`)

```text
============================== 47 passed in 11.60s =============================
```

All **47 tests pass** across `test_recommender.py` (core scoring + ranking),
`test_guardrails.py` (safety filter), `test_ai_reliability.py` (LLM-output
validation, confidence scoring, trained-model validity, end-to-end reliability), and
`test_catalog.py` (integrity of the 1,030-row catalog). Full
per-test log: [logs/tests.txt](logs/tests.txt).

---

## Portfolio Artifact

**📦 Code (GitHub):** https://github.com/Misha-te/applied-ai-system-final

**🚀 Live app (Streamlit Community Cloud):** https://share.streamlit.io/user/misha-te

**What this project says about me as an AI engineer.**
I build AI systems that are honest about their own limits. Given a small recommender, I
didn't stop at "it returns songs" — I replaced my hand-picked weights with a trained
model and then *measured* whether it actually recovers known taste (precision@5 = 0.75),
labeled every prediction with a confidence rating, wrapped the LLM in guardrails and
output validation, and wrote 47 tests that prove each claim rather than assert it. Just
as importantly, I documented where it's *weak* — 24 liked songs labeling a 1,030-song
catalog, accuracy that's only in-sample, a full-catalog precision number that looks bad
because the labels ran out — instead of hiding it. That's the engineer I want to be: someone who treats "seems to work" and "is proven
to work" as different things, who designs the safety and evaluation layers as first-class
parts of the system, and who can defend the numbers with data instead of a good story.

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
