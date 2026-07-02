import json
import os
import random
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

import db

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found. Set it in .env")

MODEL_NAME = "gemini-3.1-flash-lite" # Using a strong model for generation

class QAData(BaseModel):
    question: str = Field(description="A realistic question a user might ask regarding this theme.")
    reference_answer: str = Field(description="A comprehensive, accurate answer grounded ONLY in the provided bhashya.")

class ThemeEvalData(BaseModel):
    qa_pairs: list[QAData] = Field(description="List of question and answer pairs generated for the theme.")

def generate_dataset_for_theme(theme_name: str, verses_context: list[dict], num_questions: int = 2) -> list[dict]:
    """Generates synthetic questions and reference answers for a given theme."""
    
    # We only take up to 5 verses to avoid huge context sizes
    sample_verses = random.sample(verses_context, min(5, len(verses_context)))
    
    evidence_parts = []
    for v in sample_verses:
        evidence_parts.append(
            f"### Verse {v['verse_id']}\n"
            f"**Shloka:**\n{v['shloka']}\n"
            f"**Hindi Bhashya:**\n{v['bhashya_hindi']}\n"
        )
    evidence_block = "\n---\n".join(evidence_parts)

    system_prompt = """You are an expert on the Bhagavad Gita and Adi Shankaracharya's commentary.
You are helping to build an evaluation dataset for a chatbot.

I will provide you with a 'Theme' and some evidence (Hindi Bhashya).
Your task is to generate realistic user questions that fall exactly under this theme. 
For each question, provide a high-quality 'reference_answer' that is completely grounded in the provided Hindi Bhashya. 

The questions should be natural, what a spiritual seeker might ask.
The reference answers should be clear, accurate, and cite the verse numbers provided."""

    user_prompt = f"""## Theme: {theme_name}

## Evidence (Hindi Bhashya)
{evidence_block}

Please generate {num_questions} unique question and reference answer pairs based ON THIS EVIDENCE AND THEME."""

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )
    
    structured_llm = llm.with_structured_output(ThemeEvalData)
    
    print(f"Generating for theme: {theme_name}...")
    try:
        result: ThemeEvalData = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        
        dataset_entries = []
        for qa in result.qa_pairs:
            dataset_entries.append({
                "question": qa.question,
                "ground_truth_theme": theme_name,
                "reference_answer": qa.reference_answer
            })
        return dataset_entries
    except Exception as e:
        print(f"Error generating for {theme_name}: {e}")
        return []

def main():
    themes = db.get_all_theme_names()
    dataset = []
    
    # We can limit the number of themes for testing, or run on all
    for theme in themes:
        verse_ids = db.get_verse_ids_for_theme(theme)
        if not verse_ids:
            continue
            
        verses_context = db.get_verses_by_ids(verse_ids)
        entries = generate_dataset_for_theme(theme, verses_context, num_questions=2)
        dataset.extend(entries)
        
    output_file = "eval_dataset_draft.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\nGenerated {len(dataset)} evaluation questions.")
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    main()
