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
- Scope: a **classroom system** — but on a real catalog: **1,030 songs by 193 artists
  across 13 world regions**, pulled from an API, not typed in by hand.

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
Balanced accuracy: 86.8%   (4,120 listener–song examples, 4 labeled listeners)
Plain accuracy:    81.9%   (always-say-no baseline: 99.4%)

feature         hand-tuned   learned
genre                 0.30      0.17      ← I thought genre would dominate…
energy                0.20      0.29      ← …the data said energy matters more
acousticness          0.05      0.19      ← …and acousticness far more
```

Growing the catalog broke this model before it fixed it: with 24 likes among 4,120
examples, the unweighted trainer learned to say "no" to everything and scored a
meaningless 98%. Balanced class weights, and quoting **balanced** accuracy, is the fix.

I stopped **guessing** the weights and **learned** them, with a plain-Python logistic
regression and no ML libraries so the whole thing stays readable. The most satisfying moment
of the project was watching the data disagree with me — I had been sure genre should
dominate, and I was wrong. And I can defend the new numbers with evidence instead of a hunch.

---

## Slide 6 — Proving reliability + safety  ⏱ ~70s

> Run: `python -m src.evaluate`  and  `pytest -v`

Reliability — does it recover the songs a listener is *known* to like?
```
labeled pool (32 songs)    precision@5 = 0.75   recall@5 = 0.67   all "High" confidence
full catalog (1,030 songs) precision@5 = 0.35
```
The gap is the most honest slide in the deck. It is **not** a regression — precision counts
anything unlabeled as a miss, and my like-lists were written when the catalog was 32 songs.
The afropop listener scores 0.00 on the full catalog while their actual top five are Miriam
Makeba, Eddy Kenzo, Willy Paul and Otile Brown. The evaluation became a test of *label
coverage*; the fix is more labels, not fewer songs.

Safety — two layers I built:
- **Guardrails** block profanity and violence, but I made sure they *don't* false-flag
  "Di**ck**inson" or "**shi**itake" — matching is on whole words.
- **LLM-output validation** clamps nonsense like `energy=2.5 → 1.0` and drops invented genres
  before they can reach the scorer.

All **47 of my automated tests pass**, including a new suite that checks the integrity of
a 1,030-row catalog I can no longer read by eye. "Seems to work" and "is proven to work" are different
things, and this is where I prove it.

---

## Slide 7 — What I learned  ⏱ ~55s

- **A small dataset can fake learning.** At 32 songs, doubling a weight barely moved the
  rankings — the catalog was too clustered. At 1,030 it does move. The lesson: measure
  sensitivity, don't assume it, and suspect the data before the model.
- **Scaling the data broke two things that looked fine.** Class balance in the trainer, and
  a precision metric that silently turned into a label-coverage metric. Both were invisible
  until the catalog got 30× bigger.
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
