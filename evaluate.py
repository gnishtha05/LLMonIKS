import json
import os
import sys
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rouge_score import rouge_scorer
import statistics
# Use flush=True for prints if needed
import sys
sys.stdout.reconfigure(encoding='utf-8')
import langgraph_app

# Mock CrossEncoder to prevent loading hang since we are bypassing Node 1.5
import sys as _sys
from unittest.mock import MagicMock
_sys.modules['sentence_transformers'] = MagicMock()

import langgraph_app

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found.")

MODEL_NAME = "gemma-4-31b-it"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

class MetricScore(BaseModel):
    score: int = Field(description="An integer score from 0 to 1. Use 1 for Yes/Pass/Relevant, 0 for No/Fail/Irrelevant.")
    reasoning: str = Field(description="A brief explanation for why you gave this score.")

def calculate_answer_relevance(question: str, generated_answer: str) -> float:
    """Evaluates if the answer directly addresses the question."""
    prompt = f"""You are an expert evaluator. 
Given a question and an answer, evaluate if the answer directly and completely addresses the question.
It should not be evasive or completely off-topic.
Give a score of 1 if it is relevant, and 0 if it is irrelevant.

Question: {question}
Answer: {generated_answer}
"""
    structured_llm = llm.with_structured_output(MetricScore)
    try:
        res = structured_llm.invoke(prompt)
        return float(res.score)
    except Exception as e:
        print(f"Error scoring relevance: {e}")
        return 0.0

def calculate_faithfulness(generated_answer: str, retrieved_context: list[dict]) -> float:
    """Evaluates if the generated answer is completely grounded in the retrieved context."""
    context_str = "\n".join([f"Verse {v['verse_id']}: {v['bhashya_hindi']}" for v in retrieved_context])
    prompt = f"""You are an expert evaluator checking for hallucinations.
Given an answer and a set of retrieved contexts, evaluate if every claim made in the answer can be inferred from the context.
If the answer hallucinates or adds outside information, give a score of 0.
If the answer is completely faithful to the context, give a score of 1.

Retrieved Context:
{context_str}

Generated Answer:
{generated_answer}
"""
    structured_llm = llm.with_structured_output(MetricScore)
    try:
        res = structured_llm.invoke(prompt)
        return float(res.score)
    except Exception as e:
        print(f"Error scoring faithfulness: {e}")
        return 0.0

def calculate_lexical_accuracy(generated_answer: str, reference_answer: str) -> float:
    """Calculates ROUGE-L f-measure."""
    scores = scorer.score(reference_answer, generated_answer)
    return scores['rougeL'].fmeasure

def main():
    if not os.path.exists("eval_dataset_draft.json"):
        print("Dataset eval_dataset_draft.json not found!")
        return
        
    with open("eval_dataset_draft.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if not dataset:
        print("Dataset is empty.")
        return

    print(f"Starting evaluation on {len(dataset)} examples...")
    
    metrics = {
        "theme_accuracy": [],
        "answer_relevance": [],
        "faithfulness": [],
        "lexical_accuracy": []
    }
    
    results = []

    import time
    
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        gt_theme = item["ground_truth_theme"]
        ref_answer = item["reference_answer"]
        
        print(f"\n--- Example {i}/{len(dataset)} ---")
        print(f"Q: {question}")
        
        # 1. Run the pipeline (Bypassing Node 1.5 rank_verses as requested)
        try:
            state = {
                "question": question,
                "matched_theme": "",
                "theme_reasoning": "",
                "verse_ids": [],
                "verses_context": [],
                "answer": "",
            }
            state.update(langgraph_app.map_theme(state))
            
            # Note: We skip rank_verses! We pass the full mapped context directly to generation
            state.update(langgraph_app.generate_answer(state))
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            continue
            
        gen_answer_raw = state["answer"]
        if isinstance(gen_answer_raw, list):
            text_parts = [p.get("text", "") for p in gen_answer_raw if isinstance(p, dict) and p.get("type") == "text"]
            gen_answer = "\n".join(text_parts) if text_parts else str(gen_answer_raw)
        else:
            gen_answer = str(gen_answer_raw)
            
        matched_theme = state["matched_theme"]
        verses_context = state["verses_context"]
        
        # 2. Score Theme Accuracy
        theme_acc = 1.0 if matched_theme == gt_theme else 0.0
        
        # 3. Score Generational Metrics
        rel_score = calculate_answer_relevance(question, gen_answer)
        faith_score = calculate_faithfulness(gen_answer, verses_context)
        lex_score = calculate_lexical_accuracy(gen_answer, ref_answer)
        
        metrics["theme_accuracy"].append(theme_acc)
        metrics["answer_relevance"].append(rel_score)
        metrics["faithfulness"].append(faith_score)
        metrics["lexical_accuracy"].append(lex_score)
        
        results.append({
            "question": question,
            "gt_theme": gt_theme,
            "matched_theme": matched_theme,
            "theme_accuracy": theme_acc,
            "answer_relevance": rel_score,
            "faithfulness": faith_score,
            "lexical_accuracy": lex_score
        })
        
        print(f"Theme Acc: {theme_acc} | Relevance: {rel_score} | Faithfulness: {faith_score} | ROUGE-L: {lex_score:.3f}")
        
        # Sleep for 20 seconds to stay well below the 15 RPM limit (3 calls per loop -> ~9 RPM)
        if i < len(dataset):
            print("Sleeping for 20 seconds to respect rate limits...")
            time.sleep(20)

    # Calculate final averages
    if not results:
        print("No results calculated.")
        return
        
    avg_theme_acc = statistics.mean(metrics["theme_accuracy"])
    avg_rel = statistics.mean(metrics["answer_relevance"])
    avg_faith = statistics.mean(metrics["faithfulness"])
    avg_lex = statistics.mean(metrics["lexical_accuracy"])
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Examples: {len(results)}")
    print(f"Theme Accuracy:    {avg_theme_acc:.2%}")
    print(f"Answer Relevance:  {avg_rel:.2%}")
    print(f"Faithfulness:      {avg_faith:.2%}")
    print(f"Lexical Accuracy:  {avg_lex:.3f} (ROUGE-L)")
    
    # Save detailed results
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "averages": {
                "theme_accuracy": avg_theme_acc,
                "answer_relevance": avg_rel,
                "faithfulness": avg_faith,
                "lexical_accuracy": avg_lex
            },
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print("Saved detailed results to evaluation_report.json")

if __name__ == "__main__":
    main()
