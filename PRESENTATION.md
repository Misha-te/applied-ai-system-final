# 🎧 MuxiReco — Final Presentation

My presentation script. **8 slides, aiming for 5–7 minutes.** Each slide has the spoken
lines I'll say plus a timing box. Where I demo live, the command to run is noted; if the
terminal misbehaves I'll fall back to the saved logs in [logs/](logs/).

> Repo: https://github.com/Misha-te/applied-ai-system-final

---

## Slide 1 — Title  ⏱ ~20s

# 🎵 MuxiReco
### A transparent, content-based music recommender that learns its own weights and explains every pick.

MuxiReco takes a listener's taste and returns ranked songs — but the point isn't just that
it works, it's that every pick is explainable, measured, and safe. That's the story I want
to tell in the next six minutes.

---

## Slide 2 — The problem & why it matters  ⏱ ~40s

- Recommenders are everywhere — Spotify, TikTok — and mostly **opaque**: *"why am I seeing this?"*
- My goal was a **fully transparent** recommender: every score explainable, every weight
  readable, every bias documented rather than hidden.
- Scope: a **classroom system** on a small hand-made catalog of 32 songs — built to
  *understand* the idea, not to ship it.

I wanted to open the black box. If I can't explain and measure a recommendation, I don't
trust it — and I don't think a user should either.

---

## Slide 3 — System architecture  ⏱ ~60s

```
Listener taste ─▶ [Guardrails] ─▶ [LLM: NL → prefs] ─▶ [Validation]
                                                            │
   Catalog (songs.csv) ─▶ [load_songs] ─────────────▶ [Recommender core]
                                                            │  uses
   Labeled taste data ─▶ [train_weights.py] ─▶ learned weights ┘
                                                            │
                                        Top-k songs + score% + reasons + confidence
```

My system has five parts:
- **Guardrails** — I screen free text *before* it's ever used.
- **LLM** (Gemini/DeepSeek) — turns "something chill for studying" into preferences, and
  **degrades to a scripted flow if there's no key**, so the system never hard-depends on it.
- **Trained model** — a logistic regression that *learns* the feature weights.
- **Recommender core** — scores and ranks with a diversity penalty.
- **Tests + evaluation** — I prove it works, I don't just assert it.

It's input → process → output, with a human and a test suite checking the AI at every step.

---

## Slide 4 — Live demo 1: recommendations + explanations  ⏱ ~70s

> Run: `python -m src.main`

Looking at the **High-Energy Pop** listener:
- My top pick is **Sunrise City at 95.67%**, and it tells me why — genre matches,
  energy/valence/danceability are very close, mood is happy.
- **#2, Levitating,** shows a **−10 diversity penalty** — "genre pop already appears above."
  I built the system to spread picks out *and* to admit when it did.
- **#3–5** are non-pop songs that still fit on mood and the numeric features — honest partial
  matches instead of pretending everything is pop.

Every number comes with a sentence. That explainability was my whole design goal.

*(Backup: [logs/cli_demo.txt](logs/cli_demo.txt).)*

---

## Slide 5 — The trained model  ⏱ ~55s

> Run: `python -m src.train_weights`

```
Train accuracy: 89.1%   (128 listener–song examples, 4 labeled listeners)

feature         hand-tuned   learned
genre                 0.30      0.17      ← I thought genre would dominate…
energy                0.20      0.29      ← …the data said energy matters more
acousticness          0.05      0.17      ← …and acousticness far more
```

I stopped **guessing** the weights and **learned** them, with a plain-Python logistic
regression and no ML libraries so the whole thing stays readable. The most satisfying moment
of the project was watching the data disagree with me — I had been sure genre should
dominate, and I was wrong. And I can defend the new numbers with evidence instead of a hunch.

---

## Slide 6 — Proving reliability + safety  ⏱ ~70s

> Run: `python -m src.evaluate`  and  `pytest -v`

Reliability — does it recover the songs a listener is *known* to like?
```
MEAN   precision@5 = 0.80   recall@5 = 0.74   every top pick "High" confidence
```
I also name the weak spot honestly: the Afrobeats listener's recall is 0.44, but that's
because they have 9 liked songs — more than a top-5 list can hold — not a model failure.

Safety — two layers I built:
- **Guardrails** block profanity and violence, but I made sure they *don't* false-flag
  "Di**ck**inson" or "**shi**itake" — matching is on whole words.
- **LLM-output validation** clamps nonsense like `energy=2.5 → 1.0` and drops invented genres
  before they can reach the scorer.

All **41 of my automated tests pass**. "Seems to work" and "is proven to work" are different
things, and this is where I prove it.

---

## Slide 7 — What I learned  ⏱ ~55s

- **A small dataset can fake learning.** When I doubled a weight, the rankings barely moved —
  my catalog is too clustered. The lesson: measure sensitivity, don't assume it.
- **Validation belongs where untrusted data enters** — right at the LLM's output.
- **On working with AI as a pair-programmer:** it correctly diagnosed a misleading "no key"
  error that was really a missing package — but it also handed me dead code that passed the
  tests anyway. The lesson I'm taking away is that I own the final judgment, not the tool.

My biggest takeaway is engineering humility: build the evaluation and safety layers as
first-class parts of the system, and stay skeptical of anything — including AI — that only
*looks* finished.

---

## Slide 8 — Close  ⏱ ~30s

MuxiReco is small, but it's honest, measured, and safe.

- 🔗 Code: **github.com/Misha-te/applied-ai-system-final**
- 📊 Reproducible evidence and logs are in the README — it can be graded without a video.
- 📄 My full responsible-AI reflection is in [model_card.md](model_card.md).

That's the kind of engineer I want to be: someone who treats explainability, evaluation, and
safety as parts of the system, not extras. Thank you — I'm happy to take questions.

---

### 🎤 Timing cheat-sheet (total ≈ 6:00)

| Slide | Topic | Time |
|------:|-------|-----:|
| 1 | Title | 0:20 |
| 2 | Problem | 0:40 |
| 3 | Architecture | 1:00 |
| 4 | Demo: recommendations | 1:10 |
| 5 | Trained model | 0:55 |
| 6 | Reliability + safety | 1:10 |
| 7 | What I learned | 0:55 |
| 8 | Close | 0:30 |
| | **Total** | **~6:00** |

**Demo backup plan:** if the terminal misbehaves I'll open [logs/cli_demo.txt](logs/cli_demo.txt),
[logs/evaluate.txt](logs/evaluate.txt), and [logs/guardrails_demo.txt](logs/guardrails_demo.txt) —
they hold the exact same output.
