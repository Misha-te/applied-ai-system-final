"""
Music Recommender — conversational "DJ" Streamlit app (Free / Gemini / DeepSeek).

A guided chatbot front-end for the SAME recommendation engine the CLI uses. The
chat only *collects* a preference profile; the actual scoring/ranking is still
recommend_songs() in recommender.py — nothing is re-implemented here.

Three ways to run it, chosen from the sidebar:
  • Free (no AI) — preset choices + scripted text, no key needed.
  • Gemini       — Google's API (key in secrets as GEMINI_API_KEY).
  • DeepSeek     — OpenAI-compatible API (key in secrets as DEEPSEEK_API_KEY).

With any AI provider on, the DJ can also read FREE-TEXT answers you type and turn
them into preferences, and write a personal closing line. Without AI it still
works fully using presets.

Run it from the project root:

    streamlit run src/app.py
"""

import json
from pathlib import Path

import streamlit as st

try:  # works when launched as part of the package
    from src.recommender import load_songs, recommend_songs
    from src.guardrails import check_text
except ModuleNotFoundError:  # streamlit puts src/ on the path, so this is the usual one
    from recommender import load_songs, recommend_songs
    from guardrails import check_text

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

# --- Provider settings -----------------------------------------------------
# "gemini-flash-latest" auto-tracks Google's current flash model so it won't
# break when a specific version is retired.
GEMINI_MODEL = "gemini-flash-latest"

# DeepSeek is OpenAI-API-compatible: same SDK, just a different base_url.
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

PROVIDERS = ["Free (no AI)", "Gemini", "DeepSeek"]
PLACEHOLDER_KEYS = {"paste-your-gemini-key-here", "paste-your-deepseek-key-here", ""}

st.set_page_config(page_title="Music DJ", page_icon="🎧", layout="centered")


# ==========================================================================
# AI providers. Every path degrades gracefully to None -> scripted behavior.
# ==========================================================================
def _read_secret(name: str) -> str | None:
    """Return a configured secret value, or None if missing/placeholder."""
    try:
        value = st.secrets[name]
    except Exception:
        return None
    return None if value in PLACEHOLDER_KEYS else value


@st.cache_resource
def gemini_client():
    key = _read_secret("GEMINI_API_KEY")
    if key is None:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception as exc:
        st.session_state["ai_error"] = str(exc)
        return None


@st.cache_resource
def deepseek_client():
    key = _read_secret("DEEPSEEK_API_KEY")
    if key is None:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
    except Exception as exc:
        st.session_state["ai_error"] = str(exc)
        return None


def active_provider() -> str:
    return st.session_state.get("provider", "Free (no AI)")


def active_client():
    """The client for the currently-selected provider, or None (Free / missing key)."""
    provider = active_provider()
    if provider == "Gemini":
        return gemini_client()
    if provider == "DeepSeek":
        return deepseek_client()
    return None


def _gemini_reply(prompt: str, json_mode: bool) -> str | None:
    client = gemini_client()
    if client is None:
        return None
    try:
        config = {"response_mime_type": "application/json"} if json_mode else None
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=config
        )
        return (response.text or "").strip()
    except Exception as exc:
        st.session_state["ai_error"] = str(exc)
        return None


def _deepseek_reply(prompt: str, json_mode: bool) -> str | None:
    client = deepseek_client()
    if client is None:
        return None
    try:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        st.session_state["ai_error"] = str(exc)
        return None


def ai_reply(prompt: str, json_mode: bool = False) -> str | None:
    """Ask the selected provider for text. Returns None when AI is off or fails,
    so callers can fall back to scripted behavior."""
    provider = active_provider()
    if provider == "Gemini":
        return _gemini_reply(prompt, json_mode)
    if provider == "DeepSeek":
        return _deepseek_reply(prompt, json_mode)
    return None


def ai_extract_prefs(text: str, hint: str) -> dict:
    """Turn a free-text answer into scorer preferences via the active provider.
    Returns {} when AI is off or unsure — the scorer simply skips missing fields."""
    prompt = (
        f'A music listener answered "{hint}" with: "{text}".\n'
        "Translate that into song-search preferences. Respond with ONLY a json "
        "object, using any of these keys you are confident about and omitting "
        "the rest:\n"
        f'- "genre": one of {GENRES}\n'
        f'- "mood": one of {MOODS}\n'
        '- "energy": number 0..1 (calm=low, hype=high)\n'
        '- "valence": number 0..1 (dark/sad=low, bright/happy=high)\n'
        '- "danceability": number 0..1\n'
        '- "acousticness": number 0..1 (electronic=low, acoustic=high)\n'
        "If nothing is clear, return an empty json object {}."
    )
    raw = ai_reply(prompt, json_mode=True)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return _clean_prefs(data)


def _clean_prefs(data: dict) -> dict:
    """Keep only valid, in-range fields from an AI-produced prefs dict."""
    prefs: dict = {}
    if not isinstance(data, dict):
        return prefs
    genre = str(data.get("genre", "")).lower()
    if genre in GENRES:
        prefs["genre"] = genre
    mood = str(data.get("mood", "")).lower()
    if mood in MOODS:
        prefs["mood"] = mood
    for field in ("energy", "valence", "danceability", "acousticness"):
        try:
            value = float(data[field])
        except (KeyError, TypeError, ValueError):
            continue
        prefs[field] = max(0.0, min(1.0, value))
    return prefs


# ==========================================================================
# Catalog + conversation script.
# ==========================================================================
@st.cache_data
def get_songs():
    return load_songs(str(CSV_PATH))


songs = get_songs()
GENRES = sorted({str(s["genre"]).lower() for s in songs})
MOODS = sorted({str(s["mood"]).lower() for s in songs})

QUESTIONS = [
    {
        "key": "name",
        "question": "Hey! 🎧 I'm your DJ. First — what should I call you?",
        "options": [],
    },
    {
        "key": "mood",
        "question": "Nice to meet you! What's your mood right now?",
        "options": [
            {"label": "😊 Happy & upbeat", "prefs": {"mood": "happy", "valence": 0.85}},
            {"label": "😌 Chill & relaxed", "prefs": {"mood": "chill", "valence": 0.6}},
            {"label": "🔥 Intense & pumped", "prefs": {"mood": "intense", "valence": 0.5}},
            {"label": "💜 Romantic", "prefs": {"mood": "romantic", "valence": 0.7}},
            {"label": "🎧 Focused", "prefs": {"mood": "focused", "valence": 0.55}},
        ],
    },
    {
        "key": "energy",
        "question": "How much energy are you after?",
        "options": [
            {"label": "🌙 Low & calm", "prefs": {"energy": 0.3}},
            {"label": "🚶 In the middle", "prefs": {"energy": 0.55}},
            {"label": "⚡ High & hype", "prefs": {"energy": 0.85}},
        ],
    },
    {
        "key": "vibe",
        "question": "Bright and positive, or darker and moodier?",
        "options": [
            {"label": "☀️ Bright & positive", "prefs": {"valence": 0.85}},
            {"label": "🌗 Somewhere between", "prefs": {"valence": 0.55}},
            {"label": "🌑 Dark & moody", "prefs": {"valence": 0.25}},
        ],
    },
    {
        "key": "activity",
        "question": "What are you up to while listening?",
        "options": [
            {"label": "📚 Studying / working", "prefs": {"danceability": 0.4, "acousticness": 0.7}},
            {"label": "🏋️ Working out", "prefs": {"danceability": 0.7, "energy": 0.85}},
            {"label": "🎉 Party / hanging out", "prefs": {"danceability": 0.9}},
            {"label": "🛋️ Just relaxing", "prefs": {"danceability": 0.5}},
        ],
    },
    {
        "key": "genre",
        "question": "Any genre you're leaning toward?",
        "options": [
            {"label": "🎤 Pop", "prefs": {"genre": "pop"}},
            {"label": "🎧 Lofi", "prefs": {"genre": "lofi"}},
            {"label": "🎸 Rock", "prefs": {"genre": "rock"}},
            {"label": "🌍 Afropop", "prefs": {"genre": "afropop"}},
            {"label": "🎲 Surprise me", "prefs": {}},
        ],
    },
    {
        "key": "freeform",
        "question": "Last thing — anything else you want in the mix? Type it in your own words, or skip.",
        "options": [
            {"label": "⏭️ Skip", "prefs": {}},
        ],
    },
]


def build_prefs() -> dict:
    prefs: dict = {}
    for answer in st.session_state.answers:
        prefs.update(answer["prefs"])
    return prefs


def listener_name() -> str:
    return st.session_state.get("listener") or "friend"


def ai_extract_name(text: str) -> str | None:
    """Pull just the first name out of a free-form intro like
    "my name is Michel, how about you?" — returns None if AI is off."""
    reply = ai_reply(
        f'Extract ONLY the first name the person gives in this introduction, '
        f'and reply with just that name (no punctuation): "{text}"'
    )
    if not reply:
        return None
    name = reply.strip().strip('".!').split()
    return name[0] if name else None


def ai_dj_line(step: int) -> str | None:
    """Generate the DJ's next message: react to what the listener just said
    (answer any question they asked), then lead into the next topic. Returns
    None when no AI provider is active, so the caller uses the scripted question."""
    history = []
    for i, answer in enumerate(st.session_state.answers):
        history.append(f'DJ: {st.session_state.dj_lines.get(i, QUESTIONS[i]["question"])}')
        history.append(f'Listener: {answer["label"]}')
    goal = QUESTIONS[step]["question"]
    prompt = (
        "You are a warm, upbeat music DJ chatting one-on-one with a listener to learn "
        "their taste before recommending songs. Stay in character. Keep it to 1-2 short "
        "sentences, conversational and human.\n\n"
        "React genuinely to the listener's LAST message — if they asked you a question "
        "(e.g. 'how about you?'), actually answer it briefly — then smoothly move the "
        "chat toward the next thing you need to know.\n\n"
        "Conversation so far:\n" + "\n".join(history) + "\n\n"
        f'The next thing to find out is: "{goal}". Ask it naturally in your own words '
        "(do not repeat it verbatim, do not number it). Write ONLY the DJ's next message."
    )
    return ai_reply(prompt)


def dj_line_for(step: int) -> str:
    """The assistant's line for a question — generated once (AI if available),
    then cached in session so reruns don't regenerate it."""
    cached = st.session_state.dj_lines.get(step)
    if cached is not None:
        return cached
    # The opener (name question) stays scripted; later lines react to context.
    line = QUESTIONS[step]["question"] if step == 0 else (ai_dj_line(step) or QUESTIONS[step]["question"])
    st.session_state.dj_lines[step] = line
    return line


def dj_closing(name: str, prefs: dict) -> tuple[str, str | None]:
    """Personal closing line before the picks. Returns (text, source) where
    source is the provider that wrote it, or None when it's the scripted line."""
    llm = ai_reply(
        f"Write ONE short, upbeat sentence (no emoji lists) introducing a music "
        f"set for a listener named {name} whose preferences are {prefs}."
    )
    if llm:
        return llm, active_provider()
    genre = prefs.get("genre")
    genre_bit = f" some {genre}" if genre else " a mix"
    mood = prefs.get("mood", "your")
    scripted = f"Alright {name}, spinning up{genre_bit} for a {mood} mood — here's your set. 🎶"
    return scripted, None


def record_answer(option: dict):
    st.session_state.answers.append(option)
    st.session_state.step += 1


def record_text(step: int, text: str):
    question = QUESTIONS[step]
    if question["key"] == "name":
        prefs = {}
        st.session_state.listener = ai_extract_name(text) or text.strip()
    else:
        prefs = ai_extract_prefs(text, question["question"])
    st.session_state.answers.append({"label": text, "prefs": prefs})
    st.session_state.step += 1


def reset_chat():
    st.session_state.step = 0
    st.session_state.answers = []
    st.session_state.dj_lines = {}
    st.session_state.pop("listener", None)
    st.session_state.pop("ai_error", None)


# ==========================================================================
# UI.
# ==========================================================================
if "step" not in st.session_state:
    reset_chat()
st.session_state.setdefault("dj_lines", {})  # guard hot-reloaded old sessions

with st.sidebar:
    st.header("🎧 Music DJ")

    st.radio(
        "AI provider",
        PROVIDERS,
        key="provider",
        help="Free needs no key. Gemini/DeepSeek read free-text answers and write "
             "personal lines when their key is set in .streamlit/secrets.toml.",
    )
    provider = active_provider()
    if provider == "Free (no AI)":
        st.info("Running without AI — presets + scripted text.")
    elif active_client() is not None:
        model = GEMINI_MODEL if provider == "Gemini" else DEEPSEEK_MODEL
        st.success(f"{provider}: on ({model})")
    else:
        key_name = "GEMINI_API_KEY" if provider == "Gemini" else "DEEPSEEK_API_KEY"
        st.warning(f"{provider}: no key. Add `{key_name}` to `.streamlit/secrets.toml`.")
    if st.session_state.get("ai_error"):
        st.caption(f"⚠️ Last AI note: {st.session_state['ai_error'][:140]}")

    with st.expander("Advanced"):
        k = st.slider("How many songs?", 1, 10, 5)
        artist_penalty = st.slider("Artist variety", 0.0, 40.0, 20.0, 5.0)
        genre_penalty = st.slider("Genre variety", 0.0, 40.0, 10.0, 5.0)

    st.button("🔄 Start over", on_click=reset_chat, use_container_width=True)

st.title("🎧 Your Music DJ")

# Persistent chat box — reply in your own words on any question.
typed = st.chat_input("Reply in your own words…")

# Replay the conversation so far, using the DJ's actual (possibly AI) lines.
for i, answer in enumerate(st.session_state.answers):
    with st.chat_message("assistant"):
        st.markdown(st.session_state.dj_lines.get(i, QUESTIONS[i]["question"]))
    with st.chat_message("user"):
        st.markdown(answer["label"])

step = st.session_state.step

# --- Still chatting -> show the current line and STOP (no picks yet). --------
if step < len(QUESTIONS):
    question = QUESTIONS[step]
    with st.chat_message("assistant"):
        st.markdown(dj_line_for(step))
        options = question["options"]
        if options:
            per_row = 3 if len(options) > 4 else 2
            for start in range(0, len(options), per_row):
                row = options[start:start + per_row]
                cols = st.columns(len(row))
                for col, opt in zip(cols, row):
                    if col.button(opt["label"], key=f"opt-{step}-{opt['label']}",
                                  use_container_width=True):
                        record_answer(opt)
                        st.rerun()
        else:
            st.caption("Type your answer in the box below. 👇")

    if typed:
        verdict = check_text(typed)
        if verdict.allowed:
            record_text(step, typed)
            st.rerun()
        else:
            st.warning(
                "🎶 Let's keep it about the music. I can't work with that kind of "
                "message — mind rephrasing? (You can also tap one of the options above.)"
            )
    st.stop()

# --- All questions answered -> reveal the set. -------------------------------
name = listener_name()
prefs = build_prefs()

closing_line, closing_source = dj_closing(name, prefs)
with st.chat_message("assistant"):
    st.markdown(closing_line)
    if closing_source:
        st.caption(f"✨ written live by {closing_source}")
    else:
        st.caption("📝 scripted line (no AI provider active)")

with st.expander("What I understood about your taste"):
    st.json(prefs)

results = recommend_songs(
    prefs, songs, k=k, artist_penalty=artist_penalty, genre_penalty=genre_penalty
)

if results:
    st.bar_chart(
        {song["title"]: score for song, score, _ in results},
        color="#7c3aed",
        horizontal=True,
    )

for rank, (song, score, reasons) in enumerate(results, start=1):
    with st.container(border=True):
        st.markdown(f"### {rank}. {song['title']}")
        st.markdown(f"**{song['artist']}** · {song['genre']} · {song['mood']}")
        st.progress(min(100, int(round(score))), text=f"Match {score:.1f}%")
        with st.expander("Why this song"):
            for reason in reasons:
                st.markdown(f"- {reason}")

st.caption(f"Happy listening, {name}! Tap **Start over** in the sidebar for a new vibe.")
