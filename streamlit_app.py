"""
Streamlit UI for the Bhagavad Gita Q&A LangGraph Pipeline
─────────────────────────────────────────────────────────
Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
from langgraph_app import run, ALL_THEME_NAMES

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gita Wisdom – Ask a Question",
    page_icon="🙏",
    layout="centered",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header */
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f59e0b, #ef4444, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: #9ca3af;
    font-size: 1rem;
}

/* Answer card */
.answer-card {
    background: linear-gradient(145deg, #1e1e2e, #2a2a3d);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1rem;
    color: #e2e8f0;
    line-height: 1.75;
    font-size: 1.02rem;
}

/* Meta badge */
.meta-badge {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Verse chips container */
.verse-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.3rem;
    margin-bottom: 1.2rem;
}
.verse-chip {
    background: rgba(139, 92, 246, 0.15);
    color: #a78bfa;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* Divider */
.styled-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    margin: 1rem 0 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🙏 Gita Wisdom</h1>
    <p>Ask any question and receive an answer grounded in Shankaracharya's commentary</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ─── Input ───────────────────────────────────────────────────────────────────
user_question = st.text_input(
    "Your question",
    placeholder="e.g. How can I control my mind and senses?",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    ask_clicked = st.button("✨ Ask", use_container_width=True, type="primary")

# ─── Process & display ──────────────────────────────────────────────────────
if ask_clicked and user_question.strip():
    with st.spinner("🔍 Finding the best theme and verses…"):
        result = run(user_question.strip())

    theme = result["matched_theme"]
    verses = result["verse_ids"]
    answer_raw = result["answer"]

    # Handle stringified list output (fallback for thinking blocks)
    if isinstance(answer_raw, list):
        text_parts = [p["text"] for p in answer_raw if isinstance(p, dict) and p.get("type") == "text"]
        answer = "\n".join(text_parts) if text_parts else str(answer_raw)
    elif isinstance(answer_raw, str) and answer_raw.strip().startswith("[{") and "'type': 'thinking'" in answer_raw:
        import ast
        try:
            parsed = ast.literal_eval(answer_raw)
            text_parts = [p["text"] for p in parsed if isinstance(p, dict) and p.get("type") == "text"]
            answer = "\n".join(text_parts) if text_parts else answer_raw
        except Exception:
            answer = answer_raw
    else:
        answer = str(answer_raw)

    # Theme badge
    st.markdown(f'<span class="meta-badge">📖 Theme: {theme}</span>', unsafe_allow_html=True)

    # Verse chips
    chips_html = "".join(f'<span class="verse-chip">{v}</span>' for v in verses)
    st.markdown(f'<div class="verse-chips">{chips_html}</div>', unsafe_allow_html=True)

    # Divider
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # Answer
    st.markdown(answer)

elif ask_clicked:
    st.warning("Please enter a question first.")

# ─── Sidebar: available themes ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Available Themes")
    for i, name in enumerate(ALL_THEME_NAMES, 1):
        st.markdown(f"**{i}.** {name}")
    st.markdown("---")
    st.caption("Powered by Gemma 4 · LangGraph · Shankaracharya's Bhashya")
