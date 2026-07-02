"""
LangGraph Bhagavad Gita Q&A Pipeline
─────────────────────────────────────
Node 1   (map_theme)      : Maps user question -> best theme -> ALL verse bhashya
Node 1.5 (rank_verses)    : Cross-encoder reranker picks top 5 most relevant verses
Node 2   (generate_answer) : Generates a grounded answer from the top-ranked bhashya

Model : Gemma 4 (gemma-4-31b-it) via Google Generative AI API
Reranker : cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import json
import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# ─── Load environment ───────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found. Set it in .env or as an environment variable.")

MODEL_NAME = "gemma-4-31b-it"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_VERSES = 5

# ─── Load cross-encoder reranker once ────────────────────────────────────────
print("[Init] Loading cross-encoder reranker model...")
reranker = CrossEncoder(RERANKER_MODEL_NAME)
print("[Init] Reranker loaded.")

# ─── Load data files once at module level ────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DATA_DIR, "themes_verses.json"), "r", encoding="utf-8") as f:
    THEMES_DATA: list[dict] = json.load(f)

with open(os.path.join(DATA_DIR, "final_data.json"), "r", encoding="utf-8") as f:
    FINAL_DATA: dict = json.load(f)

# Pre-build a theme-name → verse-ids lookup for fast access
# Strip trailing whitespace / non-breaking spaces (\xa0) from theme names
THEME_TO_VERSES: dict[str, list[str]] = {
    entry["theme"].strip().replace("\xa0", ""): entry["verses"] for entry in THEMES_DATA
}

ALL_THEME_NAMES: list[str] = list(THEME_TO_VERSES.keys())


# ─── Utility: normalise verse ID ────────────────────────────────────────────
def normalise_verse_id(verse_id: str) -> str:
    """
    themes_verses.json uses zero-padded IDs like '3.09', '4.01'
    final_data.json uses unpadded IDs like '3.9', '4.1'
    This strips leading zeros from the verse part.
    """
    parts = verse_id.split(".")
    if len(parts) == 2:
        chapter = str(int(parts[0]))
        verse = str(int(parts[1]))
        return f"{chapter}.{verse}"
    return verse_id


# ─── Structured output schema for Node 1 ────────────────────────────────────
class ThemeMatch(BaseModel):
    """The best matching theme for a user question."""
    theme: str = Field(
        description="The exact theme name from the provided list that best matches the user's question."
    )
    reasoning: str = Field(
        description="A brief explanation of why this theme was selected."
    )


# ─── LangGraph state ────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    matched_theme: str
    theme_reasoning: str
    verse_ids: list[str]
    verses_context: list[dict]
    answer: str


# ─── Node 1: Theme Mapper ───────────────────────────────────────────────────
def map_theme(state: GraphState) -> dict:
    """
    Use LLM structured output to map the user question to the single best
    theme from themes_verses.json, then retrieve verse bhashya from final_data.json.
    """
    question = state["question"]

    # Build the theme list for the prompt
    theme_list_str = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(ALL_THEME_NAMES))

    prompt = f"""You are an expert on the Bhagavad Gita and Shankaracharya's commentary.

Given the following user question, select the SINGLE BEST matching theme from the list below.
You must return the theme name EXACTLY as it appears in the list.

## Available Themes
{theme_list_str}

## User Question
{question}

Pick the one theme whose associated verses would best answer this question."""

    # LLM call with structured output
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(ThemeMatch)
    result: ThemeMatch = structured_llm.invoke(prompt)

    matched_theme = result.theme.strip()

    # Fuzzy-match: if the LLM returned a theme not exactly in our list, find closest
    if matched_theme not in THEME_TO_VERSES:
        # Try case-insensitive match
        lower_map = {t.lower(): t for t in ALL_THEME_NAMES}
        matched_theme = lower_map.get(matched_theme.lower(), ALL_THEME_NAMES[0])

    # Get ALL verse IDs for the matched theme (reranker will pick the best 5)
    raw_verse_ids = THEME_TO_VERSES[matched_theme]

    # Normalise and look up each verse in final_data.json
    verses_context = []
    valid_verse_ids = []
    for vid in raw_verse_ids:
        norm_vid = normalise_verse_id(vid)
        if norm_vid in FINAL_DATA:
            verse_data = FINAL_DATA[norm_vid]
            verses_context.append({
                "verse_id": norm_vid,
                "shloka": verse_data.get("shloka", ""),
                "bhashya_hindi": verse_data.get("bhashya_hindi", ""),
            })
            valid_verse_ids.append(norm_vid)
        else:
            print(f"  [Warning] Verse {vid} (normalised: {norm_vid}) not found in final_data.json")

    print(f"\n[Node 1] Theme Mapper")
    print(f"   Matched theme : {matched_theme}")
    print(f"   Reasoning     : {result.reasoning}")
    print(f"   Verses found  : {len(valid_verse_ids)}/{len(raw_verse_ids)}")

    return {
        "matched_theme": matched_theme,
        "theme_reasoning": result.reasoning,
        "verse_ids": valid_verse_ids,
        "verses_context": verses_context,
    }


# ─── Node 1.5: Verse Reranker (Cross-Encoder) ───────────────────────────────
def rank_verses(state: GraphState) -> dict:
    """
    Use a cross-encoder reranker to score every verse in the matched theme
    against the user question and keep only the top 5 most relevant.
    """
    question = state["question"]
    verses_context = state["verses_context"]

    if len(verses_context) <= TOP_K_VERSES:
        # No need to rank if we already have 5 or fewer
        print(f"\n[Node 1.5] Verse Reranker")
        print(f"   Skipped (only {len(verses_context)} verses, no need to rank)")
        return {}

    # Build query-document pairs for the cross-encoder
    # Use bhashya_hindi as the document text (it carries the semantic meaning)
    pairs = []
    for v in verses_context:
        # Truncate bhashya to first 512 chars to keep within model limits
        doc_text = v["bhashya_hindi"][:512]
        pairs.append((question, doc_text))

    # Score all pairs
    scores = reranker.predict(pairs)

    # Pair each verse with its score and sort descending
    scored_verses = list(zip(verses_context, scores))
    scored_verses.sort(key=lambda x: x[1], reverse=True)

    # Take top K
    top_verses = [v for v, s in scored_verses[:TOP_K_VERSES]]
    top_ids = [v["verse_id"] for v in top_verses]

    print(f"\n[Node 1.5] Verse Reranker")
    print(f"   Ranked {len(verses_context)} verses, selected top {TOP_K_VERSES}")
    for v, s in scored_verses[:TOP_K_VERSES]:
        print(f"   {v['verse_id']:8s} score={s:.4f}")

    return {
        "verse_ids": top_ids,
        "verses_context": top_verses,
    }


# ─── Node 2: Answer Generator ───────────────────────────────────────────────
def generate_answer(state: GraphState) -> dict:
    """
    Generate a grounded answer to the user's question using the
    Hindi bhashya of the retrieved verses as context.
    """
    question = state["question"]
    matched_theme = state["matched_theme"]
    verses_context = state["verses_context"]

    # Build the evidence block
    evidence_parts = []
    for v in verses_context:
        evidence_parts.append(
            f"### Verse {v['verse_id']}\n"
            f"**Shloka:**\n{v['shloka']}\n\n"
            f"**Hindi Bhashya (Shankaracharya):**\n{v['bhashya_hindi']}\n"
        )
    evidence_block = "\n---\n".join(evidence_parts)

    system_prompt = """You are a helpful and accessible assistant explaining the Bhagavad Gita based on Adi Shankaracharya's commentary.

Your answer must:
- Be grounded ONLY in the Hindi bhashya evidence provided below.
- Be concise, direct, and easy to understand for a general user without prior knowledge of the Gita.
- Avoid heavy, complex Sanskrit or philosophical terminology. If you must use a specific term, explain it simply.
- Clearly cite which verse(s) support each point you make.
- Never fabricate or add claims not present in the provided bhashya.
- Keep the overall response short and practical."""

    user_prompt = f"""## Theme: {matched_theme}

## User Question
{question}

## Verse Evidence (Hindi Bhashya by Shankaracharya)
{evidence_block}

Based ONLY on the verse evidence above, provide a thorough and grounded answer to the user's question. Cite specific verses to support your points."""

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    answer = response.content

    print(f"\n[Node 2] Answer Generator")
    print(f"   Answer length : {len(answer)} chars")

    return {"answer": answer}


# ─── Build the LangGraph ────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("map_theme", map_theme)
    workflow.add_node("rank_verses", rank_verses)
    workflow.add_node("generate_answer", generate_answer)

    # Wire edges: map_theme -> rank_verses -> generate_answer -> END
    workflow.set_entry_point("map_theme")
    workflow.add_edge("map_theme", "rank_verses")
    workflow.add_edge("rank_verses", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ─── Public entry point ─────────────────────────────────────────────────────
def run(question: str) -> dict:
    """Run the full pipeline and return the final state."""
    graph = build_graph()
    initial_state: GraphState = {
        "question": question,
        "matched_theme": "",
        "theme_reasoning": "",
        "verse_ids": [],
        "verses_context": [],
        "answer": "",
    }
    final_state = graph.invoke(initial_state)
    return final_state


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
    else:
        user_question = input("Enter your question: ")

    result = run(user_question)

    print(f"\n{'═' * 60}")
    print(f"  RESULT")
    print(f"{'═' * 60}")
    print(f"\n  Theme   : {result['matched_theme']}")
    print(f"  Verses  : {', '.join(result['verse_ids'])}")
    print(f"\n{'─' * 60}")
    print(f"\n{result['answer']}")
    print(f"\n{'═' * 60}")
