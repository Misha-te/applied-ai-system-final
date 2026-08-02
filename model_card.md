# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**MusiReco 1.0** — a simple music recommender that suggests songs and explains why.

---

## 2. Intended Use  

MusiReco recommends songs by looking at a listener's taste — their favorite genre and mood,
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

Trained to **89% accuracy** on 128 `(listener, song)` examples from four labeled taste
profiles, the model shifted the weights noticeably — energy and acousticness turned out to
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
[README](README.md#fine-tuned--specialized-model-learning-the-weights) has the full write-up.

---

## 4. Data  

The catalog has **32 songs**. It started with 10, and I grew it over time to widen the range
of taste and include music I actually listen to. This same catalog is also what the trained
weight model (Section 3) learns from, paired with four labeled listeners.

The songs cover a mix of genres — pop, lofi, rock, jazz, ambient, synthwave, funk, indie
pop, and East African styles like afropop and bongo — and a range of moods such as happy,
chill, intense, relaxed, moody, and romantic.

There are still gaps: plenty of real-world styles are missing, including several kinds of
Kenyan music, so the library doesn't represent everyone's taste.

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

**Weakness I discovered during my experiments:** My recommender is surprisingly
*insensitive* to how I tune it. When I doubled the weight on energy and halved the weight
on genre, the scores every song received changed, but the actual **ranking barely moved**
— the top one or two "most-loved" songs stayed exactly where they were, and for the Chill
Lofi listener the entire top-5 order didn't change at all. In other words, the change made
the recommendations *different in number but not in order*. This happens because the
catalog is small and clustered: the best matches already agree on genre, mood, *and* the
numeric features at once, so no reasonable reweighting is enough to knock them off the top.
The bias this hides is that the system looks like it's "learning" from my weight choices
when it really isn't — a listener whose true favorite is ranked #4 would almost never see
it promoted no matter how I adjust the percentages, because the same crowd-pleasers keep
winning by default.

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
showed me the recommender is only as good as the variety in its music library.

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
- Connect it to the **Spotify API** to pull real songs and their audio features directly,
  instead of relying on a small hand-made catalog.

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
