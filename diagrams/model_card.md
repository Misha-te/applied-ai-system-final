# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**MuxiReco 1.0** — a simple music recommender that suggests songs and explains why.

---

## 2. Intended Use  

MuxiReco recommends songs by looking at a listener's taste — their favorite genre and mood,
and how energetic, positive, danceable, and acoustic they like their music. You tell it how
many songs you want back (the "top k"), and for each one it gives a match percentage and a
short list of reasons for that score.

It's useful for anyone who wants to rank songs by several factors at once instead of just
one. One assumption it makes is that **genre matters a lot** — but that isn't true for
everyone, since some listeners care more about the overall *vibe* of a song than its
official genre.

This is built for **classroom exploration**, not for real users.

---

## 3. How the Model Works  

Every song gets a score from 0% to 100% that estimates how much the listener will like it.
The system compares each song to the listener's preferences one feature at a time (genre,
mood, energy, and so on), gives each feature a "how close is it" rating, and then blends
those together using weights that say how much each feature counts. The songs with the
highest percentages rise to the top.

The main change I made from the starter version was moving away from a **points** system
(like "+2 for genre, +1 for mood") to **percentages**, so the final number always lands
between 0% and 100% and is easy to read as "how good a match this is."

### The weights are learned, not guessed (the trained model)

The one part of the recipe I no longer hand-pick is **how much each feature counts**. Those
six weights are now **learned by a small trained model** — a logistic-regression classifier
([src/train_weights.py](src/train_weights.py)) fitted to predict whether a listener likes a
song from its per-feature sub-scores. The model's coefficients (made non-negative and
normalized) become the weights, so the scoring recipe is **trained, not guessed**. This is
the project's fine-tuned / specialized-model component.

Trained to **86.8% balanced accuracy** on 4,120 `(listener, song)` examples from four
labeled taste profiles, the model shifted the weights noticeably — energy and acousticness turned out to
matter *more* than I'd assumed, genre and mood *less*:

| Feature | Hand-picked | Learned |
|---------|:-----------:|:-------:|
| genre        | 0.30 | 0.17 |
| energy       | 0.20 | 0.29 |
| valence      | 0.15 | 0.10 |
| danceability | 0.15 | 0.19 |
| mood         | 0.15 | 0.08 |
| acousticness | 0.05 | 0.17 |

Retrain with `python -m src.train_weights`; the recommender loads the result automatically
and falls back to the hand-picked recipe if it's missing.

**Honest caveat:** this is a small, self-labeled dataset, so the accuracy is *in-sample* —
it proves the training method works, not that the weights generalize to real strangers. The
README covers the trade-offs in [Design Decisions](README.md#design-decisions).

---

## 4. Data  

The catalog has **1,030 songs by 193 artists across 13 world regions**. It started as 10
invented placeholder tracks, grew to 32 by hand, and is now built by
[src/fetch_songs.py](../src/fetch_songs.py) from the **ReccoBeats API** — a free, no-key
service that still serves the Spotify-style audio features (energy, valence, danceability,
acousticness, tempo) that Spotify closed off to new apps in Nov 2024. Every audio number in
the CSV is measured, not estimated by me. This same catalog is what the trained weight model
(Section 3) learns from, paired with four labeled listeners.

Only two fields are human judgment: **genre** and **region**, which come from a curated
189-artist roster in the fetcher, since no API supplies either cleanly. **Mood** is derived
from valence and energy by a documented rule.

Regional spread: west africa 157, east africa 138, latin america 121, north america 96,
north africa & middle east 72, europe 68, southern africa 66, south asia 65, east asia 65,
caribbean 57, southeast asia 45, oceania 40, central africa 30 — plus the 10 original
invented tracks, marked `demo`.

Gaps that remain, stated plainly:

- The 10 `demo` tracks are **not real songs**. They are kept because the trainer's labels
  and the sample profiles reference them by id.
- Region labels the **artist**, not the song's musical origin, and it is one label per
  artist — a genre-crossing artist gets flattened to one tag.
- Some artists ReccoBeats can't resolve are simply missing (Beyoncé, Björk, Sigur Rós,
  Utada Hikaru, Sơn Tùng M-TP, Hoàng Thùy Linh; `IU` is too short for the API's search).
- Track selection within an artist is by ReccoBeats popularity, which skews to recent and
  streaming-heavy releases — so older catalog material is thinner than a listener would
  expect.

---

## 5. Strengths  

The system does well at what it was built for: it produces a clear ranking of songs and,
just as importantly, a plain-English reason for each pick. Listeners with a strong, clear
taste (very chill, or very high-energy) get results that match their intuition, and the
reasons make it easy to see *why* a song was chosen — which builds trust in the ranking.

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

**Weakness I discovered during my experiments:** at 32 songs my recommender was
surprisingly *insensitive* to how I tuned it. Doubling the weight on energy and halving
genre changed every score but **barely moved the ranking** — for the Chill Lofi listener
the top-5 order didn't change at all. The catalog was small and clustered, so the best
matches already agreed on genre, mood *and* the numeric features at once, and no
reasonable reweighting could knock them off. The system looked like it was "learning"
from my weight choices when it really wasn't.

Growing the catalog to 1,030 songs is what fixed that, and it exposed two further biases
worth naming:

- **Class imbalance in the trainer.** With 24 liked songs among 4,120 examples, the
  unweighted model learned to answer "no" to everything — 98% accuracy with the valence
  and danceability weights driven to zero. Balanced class weights fixed it, and the model
  card now quotes **balanced** accuracy, where always-say-no scores 0.50.
- **Label coverage, not model quality, now limits the evaluation.** See Section 7.

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.

### Profiles I tested

I built three pretend listeners with clearly different taste and ran each of them through
the system to see who they'd be matched with:

- **High-Energy Pop** — loves upbeat, happy, danceable pop.
- **Chill Lofi** — wants calm, quiet, low-energy study music.
- **Deep Intense Rock** — wants hard-hitting, high-energy rock.

For each one I looked at whether the top songs actually "felt" like something that person
would enjoy, and I read the reasons the system printed to make sure they lined up with the
listener's stated taste.

### What surprised me

The biggest surprise was how **confident and steady** the top picks were. The number-one
and number-two songs for each listener felt obviously right and never changed, even when I
adjusted the settings later. The other surprise was the **Deep Intense Rock** listener:
they got one excellent match and then a steep drop-off, because the catalog barely has any
rock, so the system was forced to offer songs from other styles that only partly fit. That
showed me the recommender is only as good as the variety in its music library — which is
exactly why the catalog was later rebuilt from an API to 1,030 songs across 13 regions.

### Measured reliability, and where it now breaks down

`python -m src.evaluate` ranks the catalog for the four labeled listeners and reports
precision/recall@5 against their known likes. It prints **two numbers**, and the gap
between them is the most honest thing in this card:

| pool | songs | mean precision@5 | mean recall@5 |
|---|---:|---:|---:|
| labeled (ids 1–32, the slice the likes describe) | 32 | 0.75 | 0.67 |
| full catalog | 1,030 | 0.35 | 0.34 |

The full-catalog number is **not** a regression. Precision is closed-world: anything not
on a like-list counts as a miss, and the like-lists were written when the catalog was 32
songs. The afropop listener scores 0.00 there while their actual top five are Miriam
Makeba, Eddy Kenzo, Willy Paul and Otile Brown — five genuinely on-taste African tracks
that nobody ever labeled. The evaluation has quietly become a measure of **label
coverage**, and the fix is more labels, not fewer songs.

### Comparing the profiles (what changed, and why it makes sense)

- **High-Energy Pop vs. Chill Lofi** — These two are near opposites, and the results proved
  it. The pop listener got bright, fast, feel-good songs, while the lofi listener got quiet,
  slow, mellow ones with almost no overlap between their lists. This makes sense because one
  asked for high energy and happiness and the other asked for low energy and calm — the two
  requests pull the system toward completely different corners of the catalog.

- **High-Energy Pop vs. Deep Intense Rock** — Both listeners want *high energy*, so at first
  I expected similar lists. Instead they got mostly different songs: the pop listener got
  cheerful, danceable tracks and the rock listener got harder, more serious ones. This makes
  sense because energy alone isn't the whole story — the mood ("happy" vs. "intense") and the
  style ("pop" vs. "rock") sent them in different directions even though their energy level
  was the same. It was a good sign that the system isn't just sorting everything by loudness.

- **Chill Lofi vs. Deep Intense Rock** — This was the sharpest contrast. The lofi listener's
  top songs were calm and acoustic, while the rock listener's were fast and forceful. It
  makes sense because these two disagree on almost every setting at once — energy, mood, and
  style — so the system had no trouble telling them apart and there was zero crossover in
  their recommendations.

---

## 8. Future Work  

- Turn it into a simple web app (for example with Streamlit) so people can pick their taste
  with sliders instead of editing code.
- **Expand the labeled likes to cover the fetched catalog.** This is now the single
  highest-value next step: with only 24 liked songs describing 1,030, the evaluation
  measures label coverage more than recommendation quality.
- Add **region** as something a listener can ask for directly ("something from West
  Africa"), rather than only a label shown on the results.

---

## 9. Personal Reflection  

Working on this showed me how much of a recommendation comes from small signals about the
listener — not just the songs you like, but the ones you replay, the ones you finish in the
first few seconds, and even the time of day or the mood you're in. All of that gets used to
guess what you'll want next.

A lot of this wasn't a total surprise, because I'd already studied how TikTok's algorithm
works, so seeing similar scoring ideas behind Spotify made sense to me. What *did* catch my
interest was the idea of **exploration** — that a good recommender will sometimes suggest a
song that doesn't match your taste on purpose, just to test your reaction and learn more
about you. It reminded me of using DJ X on  Spotify, where it kept surfacing songs I'd
never heard and I'd think, "how does it know I'd like this?" Now that I understand the
scoring and the exploration trick behind it, it feels a lot less like magic — and I can
enjoy it while actually knowing what's going on underneath.

---

## 10. Responsible AI Reflection

Building something that *works* isn't the same as building something *responsible*. This
section steps back from the code to ask what could go wrong, who it might affect, and how
honestly I can account for the tools — including AI — that helped me build it.

### 10.1 What are the limitations or biases in my system?

The technical limitations are detailed in [Section 6](#6-limitations-and-bias) (the ranking
is nearly insensitive to reweighting) and [Section 4](#4-data) (the 32-song catalog leaves
whole styles out). The **biases** worth naming plainly:

- **Catalog bias — the system can only love what it contains.** With only a handful of rock
  or East African tracks, a listener with that taste gets a strong first pick and then a
  steep drop-off into songs that only half-fit. The recommender never *says* "I don't have
  enough for you" — it presents a partly-wrong list with the same confident percentages as a
  great one. Underrepresented taste is silently served worse.
- **Label bias — the "learned" weights inherit my judgment.** The trained model
  ([Section 3](#3-how-the-model-works)) learned from *four taste profiles I wrote myself*.
  So the weights aren't an objective truth about music; they're a compression of my own
  labeling choices, and the 86.8% balanced accuracy is **in-sample** — it measures fit to my
  labels, not correctness for real strangers. This bias got *sharper* as the catalog grew:
  24 liked songs now stand in for 1,030, so the labels describe about 2% of the library.
- **Popularity / majority bias.** Because the top picks are songs that satisfy genre, mood,
  *and* the numeric features at once, a few "crowd-pleasers" win by default. A listener whose
  true favorite ranks #4 rarely sees it promoted no matter how the weights move.
- **Confidence can mislead.** A song can score 95% ("High" confidence) while only matching on
  the features the listener happened to specify — a high number is not the same as a good
  recommendation, and a trusting user might not know the difference.

### 10.2 Could my AI be misused, and how would I prevent it?

It's a small classroom project, but the misuse questions are real at any scale:

- **Manipulation dressed up as taste.** The scoring engine is neutral about *why* a song is
  promoted. The same weights that surface "songs you'll like" could be tuned to push a
  label's paid tracks and generate a plausible-sounding *reason* for each — turning the
  "explainability" feature into a tool for making advertising look like a recommendation.
  **Prevention:** keep the weights and the reasons transparent and auditable (they already
  live in a readable JSON file and are printed per pick), and never let a "sponsored" signal
  enter the score without labeling it as such.
- **Abusive free-text input.** The web chat sends user text to an LLM. Without a filter, that
  channel could be used to get the model to produce slurs or harmful content in its replies.
  **Prevention:** this is already why `src/guardrails.py` exists — untrusted text is screened
  against a blocklist *before* it's acted on, and `_clean_prefs` clamps whatever the model
  returns back into a safe, in-range set of preferences.
- **Over-trust / automation bias.** Because every pick comes with a confident percentage and
  a tidy reason, a user could take the output as more authoritative than it is.
  **Prevention:** the model card and README state up front that this is a classroom
  simulation on a tiny catalog, that the accuracy is in-sample, and that the confidence label
  is a self-rating — not a guarantee.

### 10.3 What surprised me while testing reliability?

[Section 7](#7-evaluation) covers what surprised me about the *recommendations*; testing
**reliability** with the new measurement harness ([src/evaluate.py](src/evaluate.py)) added
two more:

- **The pipeline recovers taste better than I expected, but unevenly.** Across the four
  labeled listeners the top-5 hit **mean precision 0.80** — better than I'd have guessed for
  such a small system. What surprised me was *where* it broke down: the Afrobeats listener
  scored only **0.44 recall**, not because the picks were wrong, but because that profile has
  **9 liked songs and the top-5 physically can't hold them all**. A "bad" number turned out to
  be a measurement artifact, not a model failure — a good reminder to read a metric before
  trusting it.
- **My input validation was already correct — but I couldn't prove it until I wrote the
  test.** I *believed* `_clean_prefs` rejected out-of-range model output, but "I'm pretty sure
  it works" isn't reliability. Writing a test that feeds it `energy=2.5` and an invented genre
  and watching it clamp/drop them is what turned a belief into evidence. That gap between
  *seeming* to work and *proving* it is the whole point of this part of the project.

### 10.4 My collaboration with AI on this project

I used an AI assistant (Claude) as a pair-programmer throughout — for diagnosing bugs,
scaffolding tests, and pressure-testing my reasoning. Two honest examples:

- **A genuinely helpful suggestion.** When the app showed **"Gemini: no key. Add
  GEMINI_API_KEY…"**, I assumed my key was wrong. The AI traced the message back to the code
  and pointed out that `gemini_client()` returns `None` for *two* different reasons — a
  missing key **or** a failed `import` — and the UI blamed the key for both. The real cause
  was that the `google-genai` package wasn't installed in my virtual environment. That saved
  me from re-pasting keys for an hour chasing the wrong problem, and it taught me that an
  error *message* is a guess, not a diagnosis.
- **A flawed suggestion I had to correct.** While scaffolding the reliability tests, the AI
  generated a test file that included a leftover line — `clean_prefs = pytest.importorskip`
  — that did nothing useful and would have confused anyone reading the suite later. It looked
  finished and the tests still passed, which is exactly the trap: **passing tests don't mean
  the code is clean.** I caught it on review and had it removed. More broadly, the AI's first
  instinct was to declare the Gemini bug "fixed" once the package installed, before
  accounting for the fact that the app also had to be *run through* the virtual environment —
  a reminder that I own the final judgment of whether something actually works, not the tool.

The overall lesson: AI is excellent at surfacing possibilities and writing first drafts fast,
but it's confidently wrong often enough that every suggestion has to pass through my own
testing and review before it counts.
