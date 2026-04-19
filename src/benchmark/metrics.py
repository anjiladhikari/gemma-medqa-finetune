"""
Metrics calculation module for evaluating generated text.
This script computes ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-L)
comparing model-generated answers against reference answers.
"""

import json
from statistics import mean

from rouge_score import rouge_scorer

from src.core.paths import FIRST5_RESULTS_FILE, FIRST5_METRICS_FILE


def load_results() -> list:
    with open(FIRST5_RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_rouge_scores(results: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_list = []
    rouge2_list = []
    rougeL_list = []

    for item in results:
        reference = item["reference_answer"]
        generated = item["generated_answer"]

        scores = scorer.score(reference, generated)

        rouge1_list.append(scores["rouge1"].fmeasure)
        rouge2_list.append(scores["rouge2"].fmeasure)
        rougeL_list.append(scores["rougeL"].fmeasure)

    summary = {
        "num_samples": len(results),
        "average_rouge1_f": mean(rouge1_list) if rouge1_list else 0.0,
        "average_rouge2_f": mean(rouge2_list) if rouge2_list else 0.0,
        "average_rougeL_f": mean(rougeL_list) if rougeL_list else 0.0,
    }

    return summary


def main():
    results = load_results()
    summary = calculate_rouge_scores(results)

    with open(FIRST5_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to: {FIRST5_METRICS_FILE}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()