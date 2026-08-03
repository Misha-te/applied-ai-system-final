# 🎵 MuxiReco — System Architecture

A component-level view of how the recommender is organized: the main parts, how
data flows **input → process → output**, and where **humans and automated tests**
check the AI's results. (For the detailed scoring math, see
[sketch design.mmd](sketch%20design.mmd).)

```mermaid
flowchart TD
    %% ---------------- INPUT ----------------
    subgraph INPUT["🎧 INPUT — Listener taste"]
        CLI["CLI preset profiles<br/>src/main.py"]
        CHAT["Web DJ chat answers<br/>src/app.py (Streamlit)"]
    end

    %% ---------------- AGENT (web app) ----------------
    subgraph AGENT["🤖 AGENT — Conversational DJ (optional LLM)"]
        GUARD{"Guardrails filter<br/>src/guardrails.py<br/>blocklist + LDNOOBW"}
        LLM["LLM provider<br/>Gemini / DeepSeek<br/>free text → preferences"]
        GUARD -->|clean| LLM
        GUARD -.->|blocked → ask to rephrase| CHAT
    end

    %% ---------------- RETRIEVER ----------------
    subgraph RETRIEVE["📚 RETRIEVER — Song catalog"]
        CSV[("data/songs.csv<br/>32 songs")]
        LOAD["load_songs()"]
        CSV --> LOAD
    end

    %% ---------------- SPECIALIZED MODEL (offline) ----------------
    subgraph TRAIN["🧠 SPECIALIZED MODEL — trained offline"]
        LABELS["4 labeled taste profiles<br/>(curated by human)"]
        TW["train_weights.py<br/>logistic regression<br/>128 labeled examples"]
        LW[("data/learned_weights.json<br/>feature weights")]
        LABELS --> TW --> LW
    end

    %% ---------------- RECOMMENDER CORE ----------------
    subgraph CORE["⚙️ RECOMMENDER CORE — src/recommender.py"]
        PREFS["user prefs dict<br/>build_profile → profile_to_prefs"]
        SCORE["score_song()<br/>weighted match → 0–100%"]
        RANK["recommend_songs()<br/>rank + diversity penalty → top-K"]
        PREFS --> SCORE --> RANK
    end

    %% ---------------- OUTPUT ----------------
    OUT["🏆 OUTPUT — Top-K songs<br/>score % + plain-English reasons"]

    %% ---------------- TESTING & HUMAN EVALUATION ----------------
    subgraph CHECK["✅ TESTING & HUMAN EVALUATION — checking the AI"]
        PYTEST["pytest<br/>test_recommender.py<br/>test_guardrails.py"]
        ACC["train-accuracy report<br/>89% (in-sample)"]
        HUMAN["👤 Human review<br/>reads reasons, judges 'does this fit?'<br/>model_card.md profiles"]
    end

    %% ---------------- DATA FLOW ----------------
    CLI --> PREFS
    CHAT --> GUARD
    LLM --> PREFS
    LOAD --> SCORE
    LW --> SCORE
    RANK --> OUT

    %% ---------------- CHECKS (dashed = verification) ----------------
    PYTEST -. checks .-> CORE
    PYTEST -. checks .-> GUARD
    ACC -. evaluates .-> TW
    OUT -. reviewed by .-> HUMAN
    HUMAN -. refine labels / weights .-> LABELS

    classDef model fill:#ece3fb,stroke:#8250df,color:#1c1033;
    classDef human fill:#fff3d1,stroke:#d4a72c,color:#3d2c00;
    classDef test fill:#d7f7e2,stroke:#2da44e,color:#03311a;
    class TW,LW,LABELS model;
    class HUMAN human;
    class PYTEST,ACC test;
```

## Legend — mapping to system components

| Rubric component | In this project | Files |
|------------------|-----------------|-------|
| **Retriever** | Loads the song catalog that supplies recommendation candidates | `data/songs.csv`, `load_songs()` |
| **Agent** | The conversational DJ that gathers taste and (optionally) uses an LLM to read free-text answers | `src/app.py`, optional Gemini/DeepSeek |
| **Specialized / fine-tuned model** | A trained logistic-regression model that *learns* the scoring weights from labeled data | `src/train_weights.py` → `data/learned_weights.json` |
| **Recommender core** | Scores every song against the listener and ranks the top-K with reasons | `src/recommender.py` (`score_song`, `recommend_songs`) |
| **Evaluator** | Reports the trained model's accuracy; human reads the reasons to judge fit | train-accuracy print + `model_card.md` |
| **Tester** | Automated tests over the real code paths | `tests/test_recommender.py`, `tests/test_guardrails.py` |

## Where humans / testing check the AI (the dashed arrows)

- **`pytest` → core & guardrails** — automated tests verify scoring, ranking, the diversity penalty, and that unsafe input is blocked.
- **Accuracy report → trained model** — the trainer prints in-sample accuracy (89%) so the learned weights can be sanity-checked.
- **Output → human review** — a person reads each recommendation's plain-English reasons and judges whether it actually fits the listener (documented in the model card).
- **Human → training labels** — that human judgment feeds back into the curated taste profiles the model trains on, closing the loop.

## Data flow in one line

`taste input → (guardrails + LLM) → prefs → score against catalog using learned weights → rank → top-K with reasons → checked by tests + human`
