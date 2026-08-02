# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

My version, **MusiReco 1.0**, is a content-based recommender that scores every song in the
catalog against a listener's taste (favorite genre and mood, plus how energetic, positive,
danceable, and acoustic they like their music) and returns the top `k` matches. Instead of
raw points, each song gets a **0–100% score** that reads as "how good a match this is,"
and every recommendation comes with plain-English reasons explaining the score.

It runs two ways: a **command-line demo** ([src/main.py](src/main.py)) and an interactive
**"DJ" web app** ([src/app.py](src/app.py), built with Streamlit) that chats with you to
learn your taste — optionally powered by Gemini or DeepSeek. Both share the exact same
scoring engine. See [Getting Started](#getting-started) for how to run each.

---

## How The System Works

My recommender is **content-based**: it compares each song to what the user likes and
recommends the songs that "feel" closest. What makes my version different from the
starter idea is that **I score everything in percentages instead of points.** The
assignment suggested a points system (+2 for a genre match, +1 for a mood match, etc.),
but a points total like `3.4` doesn't mean anything on its own and has no ceiling.
Instead, every song gets a single **0%–100% score that reads as "how likely this user is
to like this song."** That matches how I actually pick songs (by probability), and it
makes the number easy to explain.

> A sketch of the full data flow (Input → Loop → Ranking) lives in
> [resource/sketch design.mmd](resource/sketch%20design.mmd).

### What features does each `Song` use

I use six of the song's attributes. Two are **text/categorical** and four are **numeric**
(already scaled 0–1, so I can compare them directly):

- **genre** *(text)* – e.g. pop, lofi, afropop
- **mood** *(text)* – e.g. happy, chill, moody
- **energy** *(0–1)* – how energetic/intense the song is
- **valence** *(0–1)* – how positive or happy it sounds
- **danceability** *(0–1)* – how easy it is to dance to
- **acousticness** *(0–1)* – how acoustic vs. electronic it is

I leave out `tempo_bpm` for now because it's on a much bigger scale (60–160 BPM) and
would need to be rescaled before it could be compared fairly to the 0–1 features.

### What information does the `UserProfile` store

Each profile stores:

- **Basic user info** (a name or id)
- **A favorite genre and mood** – the categorical taste the user tells us about
- **The user's average taste** across the numeric features (average energy, valence,
  danceability, acousticness), built from the songs they've listened to. This gives one
  "ideal song" that represents their taste.
- **Songs they play repeatedly**, which count *twice* when building those averages — a
  song you replay says more about your taste than one you heard once.

### How the scorer computes a match for each song (the recipe)

This is the heart of the system, and it lives in the `score_song` function in
[src/recommender.py](src/recommender.py). For every song, I score each feature **from 0 to 1** on its own, then combine those
sub-scores into one final percentage using a **weight for each feature**. The weights add
up to 100%, so the final score is also a clean 0%–100% number.

**1. Score each feature (the 0–1 part):**

- **Categorical (genre, mood):** `1.0` if the song matches the user's favorite, else
  `0.0`. Either it's a match or it isn't.
- **Numeric (energy, valence, danceability, acousticness):** `1 − |song value − user
  target|`. Because both values are already 0–1, this is just "how close are they" — a
  perfect match is `1.0`, and the further apart they are, the closer to `0`.

**2. Combine with weights (the percentage part):**

```
score = 0.30 · genre_match      (30%)
      + 0.20 · energy_closeness  (20%)
      + 0.15 · valence_closeness (15%)
      + 0.15 · dance_closeness   (15%)
      + 0.15 · mood_match        (15%)
      + 0.05 · acoustic_closeness (5%)
```

Every term is between 0 and 1 and the weights sum to `1.00`, so `score` is always between
0 and 1 → I show it as a percentage (e.g. `0.72` → **72%**).

> **Note:** those six weight *numbers* are no longer hand-set — they're now **learned by a
> trained model** (see [Fine-Tuned / Specialized Model](#fine-tuned--specialized-model-learning-the-weights)).
> The recipe's *shape* is identical; only where the numbers come from changed.

### How the weights were *originally* assigned (now the fallback recipe)

My first version of these weights was pure judgment — the guesses below. They're still in
the code as a **fallback** (used only when no trained weights file is present), but the
weights the recommender actually runs on are now **learned from data** — see
[Fine-Tuned / Specialized Model](#fine-tuned--specialized-model-learning-the-weights) for
how, and for how much the learned numbers differ from these guesses.

| Feature       | Weight | Why I *guessed* that weight                                        |
|---------------|:------:|--------------------------------------------------------------------|
| Genre         | 30%    | The strongest signal — a pop fan should mostly get pop.            |
| Energy        | 20%    | The audio feature that most changes how a song *feels*.           |
| Valence       | 15%    | Positivity matters, but less than raw energy.                     |
| Danceability  | 15%    | Same idea — a meaningful part of "feel," not the top of it.       |
| Mood          | 15%    | Kept at half of genre, matching the assignment's 2:1 genre-to-mood hint. |
| Acousticness  | 5%     | The least important cue, so it only nudges ties.                  |

The four numeric "feel" features add up to 55%, which is what lets a song still score,
say, 65% even when its genre is different — that's how the system can suggest something
outside your usual genre.

### How the code is organized (data, functions, and the profile bridge)

The **backbone of the system is a small set of functions** in
[src/recommender.py](src/recommender.py), and they all pass around **plain dictionaries**:

- `load_songs()` reads the CSV catalog into a list of song dicts.
- `score_song()` scores one song against a preference dict (the recipe above).
- `recommend_songs()` scores the whole catalog, applies a diversity penalty so the list
  doesn't pile up on one artist or genre, and returns the ranked top `k`.

On top of that I keep two small **dataclasses**, `Song` and `UserProfile`, to represent a
song and a listener's *learned taste*. These aren't a parallel copy of the logic — they
feed into it through two bridge functions:

- `build_profile()` turns a listening history (the songs a user liked, with replays
  counted twice) into a `UserProfile` of average energy / valence / danceability.
- `profile_to_prefs()` translates that `UserProfile` into the same preference dict
  `score_song()` reads, so a profile learned from history flows straight into the scorer.

So there's **one pipeline**, not two: `build_profile → profile_to_prefs → recommend_songs`.
The CLI ([src/main.py](src/main.py)) demonstrates both entry points — hand-written
preference dicts *and* a profile learned from listening history.

### How do I choose which songs to recommend

- **Mostly by probability:** I sort every song by its percentage score and recommend the
  **top K** (default 5). High score = strong match.
- **Sometimes I explore:** every so often I recommend a song with a *low* probability
  (around 10%) on purpose. This helps the user discover something new, and it gives the
  system more information about whether their taste is wider than we thought.
- **Similar-user idea (stretch goal):** if two users have very similar taste and one of
  them liked a song the other hasn't heard, I can recommend that song to the second user,
  since people with similar taste tend to like similar songs. *(This one goes beyond
  pure content-based recommending — it's a "collaborative filtering" idea I'd add later.)*

### Potential bias from how the percentages were assigned

Since the weights are now **learned** (see [Fine-Tuned / Specialized Model](#fine-tuned--specialized-model-learning-the-weights)),
the bias has *moved* rather than disappeared: the model no longer bakes in my guesses, but
it does inherit whatever bias is in the **labels I trained it on** (which listeners I
invented, and which songs I decided they "like"). The original hand-picked recipe below
still ships as a fallback, and its assumptions are worth spelling out because they show what
kind of bias a hand-set recipe *can* carry:

- **Over-weighting genre (30%).** Genre is the single heaviest feature, so the recommender
  leans hard toward the user's stated favorite genre. That can trap the user in a bubble
  (a pop fan almost never sees rock or jazz) and it's unfair to good songs that just
  happen to be labeled a different genre. It also trusts the CSV's genre labels
  completely, even though genre labels are fuzzy and inconsistent in real data.
- **Under-weighting acousticness (5%).** At only 5%, acousticness almost never changes a
  ranking, so a listener who specifically cares about acoustic vs. electronic sound is
  barely served by the system.
- **The gaps between weights are themselves a bias.** Nothing about the *data* says genre
  should be 6× more important than acousticness — that ratio is a choice I made, and a
  different, equally reasonable choice would produce different recommendations. I test
  this in the **Experiments** section below by changing the genre weight and watching the
  results shift.

---

## Fine-Tuned / Specialized Model: Learning the Weights

Everything above describes the *shape* of the scoring recipe. The one thing I no longer
hand-pick is **how much each feature counts** — those six weights are now **learned from
data by a small trained model**. This is the project's "fine-tuned / specialized model"
component: a scoring model specialized for music-taste matching whose parameters come from
training, not guesswork.

### What the model is

A **logistic-regression classifier** ([src/train_weights.py](src/train_weights.py)) trained
to answer one question: *given a song's six per-feature sub-scores for a listener, will that
listener like the song?* Once trained, the model's coefficients say how strongly each
feature pushes a song toward "liked" — and those coefficients (made non-negative and
normalized to sum to 1) become the weights `score_song` uses.

It's written in **plain Python** — a few dozen lines of gradient descent, no scikit-learn or
other ML library — so the whole pipeline is transparent and reproducible: the weights start
at zero and there's no randomness, so the same data always trains the same weights.

### How it's trained

- **Training data:** four labeled "taste profiles" (upbeat pop, chill lofi, intense rock,
  afrobeats/afropop), each with a curated set of songs they're known to like. Every profile
  is paired with all 32 songs, giving **128 `(listener, song)` examples** (24 likes, 104
  non-likes).
- **Features:** for each pair, the same six 0–1 sub-scores `score_song` computes (genre
  match, energy closeness, …).
- **Label:** `1` if the listener likes that song, else `0`.
- **Fit:** batch gradient descent with a separate bias term (so the six weights reflect
  feature importance, not the fact that most songs are non-likes) and a small L2 penalty.

Retrain any time with:

```bash
python -m src.train_weights
```

It writes [data/learned_weights.json](data/learned_weights.json) and prints a
hand-tuned-vs-learned comparison. `recommender.py` loads that file automatically on the next
import; if it's missing, the original hand-tuned recipe is used as a fallback so the app
always runs.

### What the model learned (and how it differs from my guesses)

Trained to **89% accuracy** on the 128 examples, the model disagreed with my original
hand-picked weights in ways I can defend:

| Feature       | My guess | **Learned** | What the data said |
|---------------|:--------:|:-----------:|--------------------|
| Genre         | 0.30     | **0.17**    | Weaker than I assumed — my afrobeats listener likes *bongo* and *afrobeats* tracks that don't exactly match the "afropop" genre label, so exact-genre matching earned less trust. |
| Energy        | 0.20     | **0.29**    | The single strongest feature — energy separates likes from non-likes better than genre. |
| Danceability  | 0.15     | **0.19**    | A bit more important than I'd weighted it. |
| Acousticness  | 0.05     | **0.17**    | The biggest surprise: I called it a tie-breaker, but it's a real signal (chill listeners want acoustic, rock listeners don't). |
| Valence       | 0.15     | **0.10**    | Slightly less important on its own. |
| Mood          | 0.15     | **0.08**    | Weakest — mood labels overlap a lot with genre and energy, so they add little on top. |

This directly answers a weakness I'd flagged: the weights used to be *my* judgment calls,
and I had no principled way to justify "genre is 6× acousticness." Now the ratio comes from
data.

### Honest limits of the trained model

This demonstrates the **method**, not a production model. It's a **small dataset** (4
listeners, 32 songs, labels I curated myself), so the 89% is **in-sample** — it shows the
training works, not that it generalizes to strangers. The labels also carry my own
assumptions about who likes what, so the model can only be as fair as the taste profiles I
wrote. With a real listening-history dataset (e.g. from the Spotify API — see Future Work in
the [model card](model_card.md#8-future-work)) the exact same trainer would produce far more
trustworthy weights.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the command-line app:

```bash
python -m src.main
```

### Web app (conversational DJ)

There's also a browser version built with **Streamlit** ([src/app.py](src/app.py)) — a
guided **"DJ" chatbot**. It asks your name, then walks you through a short conversation
(mood, energy, bright-vs-moody, activity, genre) and only *then* reveals its picks,
greeting you by name. It calls the **same** `recommend_songs` engine as the CLI, so none
of the scoring is duplicated. Run it from the project root:

```bash
streamlit run src/app.py
```

For every question you can either **tap a preset** or **type your own answer** in the chat
box.

#### Choosing an AI provider (Free / Gemini / DeepSeek)

A sidebar selector picks how the DJ runs:

| Provider | Needs a key? | What it adds |
|----------|:------------:|--------------|
| **Free (no AI)** | No | Preset buttons + fixed scripted questions. Works out of the box. |
| **Gemini** | `GEMINI_API_KEY` | Google's API. |
| **DeepSeek** | `DEEPSEEK_API_KEY` | OpenAI-compatible API (via the `openai` SDK + a custom `base_url`). |

With **any AI provider on**, the DJ becomes genuinely interactive. It can:

- **Read free-text answers** ("something moody for a rainy night") and turn them into the
  same preference fields the scorer uses.
- **React to what you say** — its questions are generated from the live conversation, so if
  you reply *"my name is Michel, how about you?"* it answers you and moves on, instead of
  ignoring it.
- **Write a personalized closing line**, tagged with which provider produced it.

Important: **the ranked song list is the same across providers** when you only tap presets —
the provider changes the *wording* and *free-text understanding*, never the scoring engine.
To see recommendations differ, type a free-text answer (Free mode ignores it; Gemini/DeepSeek
interpret it).

**Keys** go in `.streamlit/secrets.toml` (gitignored — never commit them). You only need a
key for the provider you plan to use:

```toml
GEMINI_API_KEY   = "your-gemini-key"     # https://aistudio.google.com/app/apikey
DEEPSEEK_API_KEY = "your-deepseek-key"   # https://platform.deepseek.com/api_keys
```

The sidebar shows the active provider's status (**on** with the model name, or a "no key"
hint).

#### Safety guardrails

Because the chat accepts free text that gets sent to an LLM, typed input is screened
before it's used ([src/guardrails.py](src/guardrails.py)). Two layers back this:

1. A curated, categorized blocklist ([data/guardrails.json](data/guardrails.json)) —
   profanity, slurs, sexual/violent content, harassment, illegal requests.
2. The bundled open-source **LDNOOBW** wordlist ([data/bad_words.txt](data/bad_words.txt))
   for broad coverage.

Matching uses **word boundaries** for single words (so "Dickinson" and "shiitake" are fine)
and substring matching for phrases ("kill yourself"). Blocked input isn't stored or sent to
the model — the DJ just asks you to rephrase. Preset buttons are inherently safe and aren't
screened.

### Running Tests

Run the test suite with:

```bash
pytest
```

The tests exercise the **real** code paths, not stubs:

- `tests/test_recommender.py` — `score_song`, `recommend_songs` (ranking **and** the
  diversity penalty), `load_songs`, and the `build_profile → profile_to_prefs` bridge.
- `tests/test_guardrails.py` — clean text passes, each blocklist category is caught, the
  word-boundary false-positive guard holds, and the bundled wordlist layer is active.

---

## Sample Recommendation Output

How the output is produced:

- Each song is scored with **weighted preference matching** — every feature the user
  cares about contributes a 0–1 sub-score, combined using the percentage weights in the
  recipe above.
- Recommendations are **ranked from highest score to lowest**.
- Only the **top `k`** results are returned (default 5).
- Each recommendation includes **reasons** explaining why it scored the way it did.

Running `python -m src.main` scores every song for three hand-written sample listeners
(shown below) and then prints one more set for a **profile learned from listening
history** via `build_profile`, to demonstrate the OOP-to-scorer bridge end to end:

```text
Top Recommendations — High-Energy Pop
=====================================
Profile: genre=pop, mood=happy, energy=0.9, danceability=0.85, valence=0.85

1. Sunrise City
   Score: 97.21%
   Reasons:
   - Genre matches your preference for pop
   - Energy is very close to your preferred level
   - Valence is very close to your preferred level
   - Danceability is very close to your preferred level
   - Mood matches your preference for happy

2. Shake It Off
   Score: 93.32%
   Reasons:
   - Genre matches your preference for pop
   - Energy is very close to your preferred level
   - Valence is very close to your preferred level
   - Danceability is close to your preferred level
   - Mood matches your preference for happy

3. Gym Hero
   Score: 81.84%
   Reasons:
   - Genre matches your preference for pop
   - Energy is very close to your preferred level
   - Valence is very close to your preferred level
   - Danceability is very close to your preferred level
   - Mood (intense) differs from your preferred happy

4. Anti-Hero
   Score: 69.95%
   Reasons:
   - Genre matches your preference for pop
   - Energy is lower than your preferred level
   - Valence is lower than your preferred level
   - Danceability is close to your preferred level
   - Mood (moody) differs from your preferred happy

5. Sura Yako
   Score: 65.16%
   Reasons:
   - Genre (afropop) differs from your preferred pop
   - Energy is close to your preferred level
   - Valence is very close to your preferred level
   - Danceability is very close to your preferred level
   - Mood matches your preference for happy


Top Recommendations — Chill Lofi
================================
Profile: genre=lofi, mood=chill, energy=0.35, danceability=0.55, acousticness=0.8

1. Library Rain
   Score: 99.12%
   Reasons:
   - Genre matches your preference for lofi
   - Energy is very close to your preferred level
   - Danceability is very close to your preferred level
   - Mood matches your preference for chill
   - Acousticness is very close to your preferred level

2. Midnight Coding
   Score: 96.59%
   Reasons:
   - Genre matches your preference for lofi
   - Energy is very close to your preferred level
   - Danceability is very close to your preferred level
   - Mood matches your preference for chill
   - Acousticness is very close to your preferred level

3. Focus Flow
   Score: 80.18%
   Reasons:
   - Genre matches your preference for lofi
   - Energy is very close to your preferred level
   - Danceability is very close to your preferred level
   - Mood (focused) differs from your preferred chill
   - Acousticness is very close to your preferred level

4. Spacewalk Thoughts
   Score: 59.88%
   Reasons:
   - Genre (ambient) differs from your preferred lofi
   - Energy is very close to your preferred level
   - Danceability is close to your preferred level
   - Mood matches your preference for chill
   - Acousticness is close to your preferred level

5. Malaika
   Score: 55.00%
   Reasons:
   - Genre (afropop) differs from your preferred lofi
   - Energy is close to your preferred level
   - Danceability is close to your preferred level
   - Mood matches your preference for chill
   - Acousticness is lower than your preferred level


Top Recommendations — Deep Intense Rock
=======================================
Profile: genre=rock, mood=intense, energy=0.9, danceability=0.6, valence=0.45

1. Storm Runner
   Score: 98.37%
   Reasons:
   - Genre matches your preference for rock
   - Energy is very close to your preferred level
   - Valence is very close to your preferred level
   - Danceability is very close to your preferred level
   - Mood matches your preference for intense

2. Gym Hero
   Score: 58.32%
   Reasons:
   - Genre (pop) differs from your preferred rock
   - Energy is very close to your preferred level
   - Valence is higher than your preferred level
   - Danceability is higher than your preferred level
   - Mood matches your preference for intense

3. Night Drive Loop
   Score: 46.79%
   Reasons:
   - Genre (synthwave) differs from your preferred rock
   - Energy is close to your preferred level
   - Valence is very close to your preferred level
   - Danceability is close to your preferred level
   - Mood (moody) differs from your preferred intense

4. Anti-Hero
   Score: 44.84%
   Reasons:
   - Genre (pop) differs from your preferred rock
   - Energy is lower than your preferred level
   - Valence is very close to your preferred level
   - Danceability is very close to your preferred level
   - Mood (moody) differs from your preferred intense

5. 24K Magic
   Score: 42.79%
   Reasons:
   - Genre (funk) differs from your preferred rock
   - Energy is very close to your preferred level
   - Valence is higher than your preferred level
   - Danceability is close to your preferred level
   - Mood (happy) differs from your preferred intense
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

- **Weight shift (sensitivity test).** I doubled the weight on energy (0.20 → 0.40) and
  halved the weight on genre (0.30 → 0.15). Every song's percentage changed, but the
  **ranking barely moved** — the top one or two songs stayed exactly the same for every
  listener, and the Chill Lofi listener's whole top-5 order didn't change at all. The
  change made the results *different in number but not in order*.
- **Adversarial / edge-case profiles.** I tried to "trick" the scorer with conflicting
  preferences (high energy + a sad mood), a genre that isn't in the catalog, out-of-range
  values like `energy = 2.0`, and profiles with only one feature set. These showed the
  scorer treats each feature independently (so it can't spot contradictions) and doesn't
  reject bad input — useful for understanding where it breaks.
- **Different user types.** I ran three profiles (High-Energy Pop, Chill Lofi, Deep Intense
  Rock) and compared their outputs — see the Sample Recommendation Output above and the
  Evaluation section of the model card.

---

## Limitations and Risks

- **Tiny catalog.** Only 32 songs, so some listeners (like the rock fan) get one good match
  and then a steep drop-off into songs that only partly fit — and it's also why the trained
  model above is trained on so few examples.
- **Over-weighting genre.** Genre is the heaviest feature, so the system leans hard toward a
  listener's stated genre and can trap them in a "bubble," while good songs in other genres
  are held back.
- **Insensitive to tuning.** Because the catalog is small and clustered, changing the weights
  mostly changes the percentages, not the ranking — the same crowd-pleasers keep winning.
- **No understanding of lyrics or language**, and several real-world styles are missing
  (including some Kenyan music), so the library doesn't represent everyone's taste.
- **No input checking**, so contradictory or out-of-range preferences still produce a
  confident-looking result.

I go deeper on these in the [model card](model_card.md).

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

The biggest thing I learned is that most of these systems turn data into predictions using
**probability**. Instead of a flat "yes or no," the recommender estimates a percentage — a
likelihood that you'll enjoy a song — and ranks by that. Two ideas built on top of this
stood out to me. The first is **exploration**: a good recommender will sometimes hand you a
song that *doesn't* match your usual taste on purpose, just to see whether you like it and
learn more about you. The second is **collaborative filtering**, which is about people
instead of song features: if person A likes songs 1, 2, 3, and 4, and person B likes songs
1, 2, and 3, then the system guesses A and B are the same "type" and recommends song 4 to
B. Done across millions and millions of users — the kind of scale Spotify and TikTok have —
that huge amount of data is exactly what makes their predictions feel so accurate.

That scale is also where **bias and unfairness** creep in. When the system leans on what's
popular with the crowd, songs that already have lots of listeners keep getting pushed, while
niche artists and less-represented styles (like many African genres) rarely get surfaced —
so the "rich get richer." Collaborative filtering can also box people in: if it decides
you're a certain "type," it mostly feeds you more of the same and quietly narrows what you
ever get to discover. My own tiny version shows a smaller version of the same problem — it
over-trusts genre and keeps recommending the same crowd-pleasers — which made it easier to
see how these risks would scale up in a real app.

---

## Credits

- Profanity/slur coverage in the web app's guardrails uses the **LDNOOBW** wordlist,
  *"List of Dirty, Naughty, Obscene, and Otherwise Bad Words"*
  (<https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words>),
  bundled at [data/bad_words.txt](data/bad_words.txt).
- Optional AI providers: **Google Gemini** (via `google-genai`) and **DeepSeek** (via the
  OpenAI-compatible `openai` SDK).



