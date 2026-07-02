# Gita Wisdom – Bhagavad Gita Q&A Pipeline

A LangGraph-powered question-answering system grounded in **Adi Shankaracharya's Hindi Bhashya** on the Bhagavad Gita. Ask any life question and receive an answer backed by the most relevant verses, selected by an intelligent multi-stage pipeline.

## Architecture

The application uses a **3-node LangGraph pipeline** with a cross-encoder reranker for intelligent verse selection:

```
User Question
     │
     ▼
┌─────────────────────────┐
│  Node 1: Theme Mapper   │  LLM (Gemma 4) with structured output
│  Maps question to best  │  picks the single best theme from 13
│  Gita theme             │  predefined categories
└────────────┬────────────┘
             │ All verses for the matched theme
             ▼
┌─────────────────────────┐
│  Node 1.5: Verse Ranker │  Cross-Encoder (ms-marco-MiniLM)
│  Scores every verse     │  scores each verse's Hindi bhashya
│  against the question   │  against the user query and picks
│  and selects top 5      │  the top 5 most relevant
└────────────┬────────────┘
             │ Top 5 ranked verses + bhashya
             ▼
┌─────────────────────────┐
│  Node 2: Answer Gen     │  LLM (Gemma 4) generates a concise,
│  Generates grounded     │  accessible answer citing specific
│  answer from bhashya    │  verses as evidence
└────────────┬────────────┘
             │
             ▼
        Final Answer
```

## Features

- **Theme Mapping** – Uses LLM structured output (Pydantic schema) to classify user questions into one of 13 Bhagavad Gita themes
- **Cross-Encoder Reranking** – Scores all theme verses against the user query to select the 5 most relevant (not just the first 5)
- **Grounded Answers** – Responses are strictly based on Shankaracharya's Hindi commentary, with verse citations
- **Accessible Language** – Answers avoid heavy Sanskrit terminology and are written for a general audience
- **Streamlit UI** – Clean web interface with theme badges, verse chips, and markdown-rendered answers

## Tech Stack

| Component         | Technology                                      |
|-------------------|-------------------------------------------------|
| Orchestration     | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM               | Gemma 4 31B IT via Google Generative AI API     |
| Reranker          | `cross-encoder/ms-marco-MiniLM-L-6-v2`         |
| LLM Integration   | `langchain-google-genai`                        |
| UI                | Streamlit                                       |
| Data              | Custom JSON datasets (Shankaracharya's Bhashya) |

## Project Structure

```
ugp/
├── langgraph_app.py       # Core pipeline: theme mapper → verse ranker → answer generator
├── streamlit_app.py       # Streamlit web UI
├── themes_verses.json     # 13 themes mapped to Gita verse IDs
├── final_data.json        # Verse data: shloka, Sanskrit bhashya, Hindi bhashya
├── .env                   # API key (GOOGLE_API_KEY)
└── README.md
```

## Data Files

### `themes_verses.json`
A curated list of 13 Bhagavad Gita themes, each containing a list of relevant verse IDs:

| # | Theme |
|---|-------|
| 1 | Bhakti/Ananya Bhakti |
| 2 | Characteristics of a Self-Realized Person |
| 3 | Defeating Impersonalism |
| 4 | Demigod Worship |
| 5 | Devotees and Non-Devotees |
| 6 | Levels of God-Realization |
| 7 | Levels of Knowledge / How to Attain Knowledge |
| 8 | Mind and Sense Control |
| 9 | Relationship between Jiva, Isvara and Prakriti |
| 10 | Soul and Transmigration |
| 11 | The Yoga Processes / Renunciation of Work vs. Work in Devotion |
| 12 | Three Modes of Material Nature |
| 13 | Varnasrama |

### `final_data.json`
Contains every Gita verse with three fields:
- `shloka` – The Sanskrit verse
- `bhashya_skt` – Shankaracharya's commentary in Sanskrit
- `bhashya_hindi` – Shankaracharya's commentary in Hindi

## Setup

### Prerequisites
- Python 3.10+
- A Google Generative AI API key ([Get one here](https://aistudio.google.com/apikey))

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ugp
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install langgraph langchain-google-genai python-dotenv sentence-transformers streamlit
   ```

4. **Set your API key:**
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Streamlit Web App (Recommended)
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501 in your browser.



## How It Works

1. **You ask a question** – e.g., *"What should I do when I feel depressed?"*

2. **Theme Mapper (Node 1)** – The LLM reads your question and selects the most relevant theme (e.g., *"Mind and Sense Control"*) using structured output to ensure a clean, exact theme name.

3. **Verse Ranker (Node 1.5)** – All verses (e.g., 12 verses) from that theme are loaded. The cross-encoder reranker scores each verse's Hindi bhashya against your question and selects the **top 5 most relevant**.

4. **Answer Generator (Node 2)** – The LLM receives your question along with the Hindi bhashya of the top 5 verses and generates a concise, accessible answer with verse citations.



