"""
Full benchmark metrics calculation module.

This script reads generated benchmark results and computes summary-only metrics:
- Exact Match
- BLEU
- ROUGE-1 / ROUGE-2 / ROUGE-L (recall, precision, F1)
- METEOR
- BERTScore
- chrF
- Semantic Similarity
- Average generated/reference answer lengths

It saves only the final averaged summary to disk.
"""

import json
import re
from statistics import mean

from bert_score import score as bertscore_score
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import TOP_K_QUESTIONS
from src.core.paths import (
    FIRST5_RESULTS_FILE,
    FIRST5_METRICS_FILE,
    FULL_RESULTS_FILE,
    FULL_METRICS_FILE,
)


RESULTS_FILE = FULL_RESULTS_FILE if TOP_K_QUESTIONS is None else FIRST5_RESULTS_FILE
METRICS_FILE = FULL_METRICS_FILE if TOP_K_QUESTIONS is None else FIRST5_METRICS_FILE


def load_results() -> list:
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(reference: str, generated: str) -> int:
    return int(normalize_text(reference) == normalize_text(generated))


def compute_bleu(reference: str, generated: str) -> float:
    ref_tokens = word_tokenize(reference)
    gen_tokens = word_tokenize(generated)

    if not ref_tokens or not gen_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)


def compute_rouge_scores(reference: str, generated: str, scorer) -> dict:
    scores = scorer.score(reference, generated)

    return {
        "rouge1_recall": scores["rouge1"].recall,
        "rouge1_precision": scores["rouge1"].precision,
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_recall": scores["rouge2"].recall,
        "rouge2_precision": scores["rouge2"].precision,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_recall": scores["rougeL"].recall,
        "rougeL_precision": scores["rougeL"].precision,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def compute_meteor(reference: str, generated: str) -> float:
    ref_tokens = word_tokenize(reference)
    gen_tokens = word_tokenize(generated)

    if not ref_tokens or not gen_tokens:
        return 0.0

    return meteor_score([ref_tokens], gen_tokens)


def compute_chrf(reference: str, generated: str, chrf_metric) -> float:
    return chrf_metric.sentence_score(generated, [reference]).score


def compute_average_bertscore(references: list[str], generated_answers: list[str]) -> float:
    if not references or not generated_answers:
        return 0.0

    _, _, f1 = bertscore_score(
        generated_answers,
        references,
        lang="en",
        verbose=False,
    )
    return f1.mean().item()


def compute_average_semantic_similarity(references: list[str], generated_answers: list[str]) -> float:
    if not references or not generated_answers:
        return 0.0

    model = SentenceTransformer("all-MiniLM-L6-v2")

    ref_embeddings = model.encode(references, convert_to_tensor=False)
    gen_embeddings = model.encode(generated_answers, convert_to_tensor=False)

    similarities = []
    for ref_emb, gen_emb in zip(ref_embeddings, gen_embeddings):
        sim = cosine_similarity([ref_emb], [gen_emb])[0][0]
        similarities.append(float(sim))

    return mean(similarities) if similarities else 0.0


def calculate_summary_metrics(results: list) -> dict:
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    chrf_metric = CHRF()

    exact_match_scores = []
    bleu_scores = []

    rouge1_recall_scores = []
    rouge1_precision_scores = []
    rouge1_f1_scores = []

    rouge2_recall_scores = []
    rouge2_precision_scores = []
    rouge2_f1_scores = []

    rougeL_recall_scores = []
    rougeL_precision_scores = []
    rougeL_f1_scores = []

    meteor_scores = []
    chrf_scores = []

    generated_lengths = []
    reference_lengths = []

    references = []
    generated_answers = []

    for item in results:
        reference = item["reference_answer"]
        generated = item["generated_answer"]

        references.append(reference)
        generated_answers.append(generated)

        exact_match_scores.append(exact_match(reference, generated))
        bleu_scores.append(compute_bleu(reference, generated))

        rouge_scores = compute_rouge_scores(reference, generated, rouge)
        rouge1_recall_scores.append(rouge_scores["rouge1_recall"])
        rouge1_precision_scores.append(rouge_scores["rouge1_precision"])
        rouge1_f1_scores.append(rouge_scores["rouge1_f1"])

        rouge2_recall_scores.append(rouge_scores["rouge2_recall"])
        rouge2_precision_scores.append(rouge_scores["rouge2_precision"])
        rouge2_f1_scores.append(rouge_scores["rouge2_f1"])

        rougeL_recall_scores.append(rouge_scores["rougeL_recall"])
        rougeL_precision_scores.append(rouge_scores["rougeL_precision"])
        rougeL_f1_scores.append(rouge_scores["rougeL_f1"])

        meteor_scores.append(compute_meteor(reference, generated))
        chrf_scores.append(compute_chrf(reference, generated, chrf_metric))

        generated_lengths.append(len(word_tokenize(generated)))
        reference_lengths.append(len(word_tokenize(reference)))

    average_bertscore = compute_average_bertscore(references, generated_answers)
    average_semantic_similarity = compute_average_semantic_similarity(references, generated_answers)

    summary = {
        "num_samples": len(results),

        "exact_match_rate": mean(exact_match_scores) if exact_match_scores else 0.0,
        "average_bleu": mean(bleu_scores) if bleu_scores else 0.0,

        "average_rouge1_recall": mean(rouge1_recall_scores) if rouge1_recall_scores else 0.0,
        "average_rouge1_precision": mean(rouge1_precision_scores) if rouge1_precision_scores else 0.0,
        "average_rouge1_f1": mean(rouge1_f1_scores) if rouge1_f1_scores else 0.0,

        "average_rouge2_recall": mean(rouge2_recall_scores) if rouge2_recall_scores else 0.0,
        "average_rouge2_precision": mean(rouge2_precision_scores) if rouge2_precision_scores else 0.0,
        "average_rouge2_f1": mean(rouge2_f1_scores) if rouge2_f1_scores else 0.0,

        "average_rougeL_recall": mean(rougeL_recall_scores) if rougeL_recall_scores else 0.0,
        "average_rougeL_precision": mean(rougeL_precision_scores) if rougeL_precision_scores else 0.0,
        "average_rougeL_f1": mean(rougeL_f1_scores) if rougeL_f1_scores else 0.0,

        "average_meteor": mean(meteor_scores) if meteor_scores else 0.0,
        "average_bertscore": average_bertscore,
        "average_chrf": mean(chrf_scores) if chrf_scores else 0.0,
        "average_semantic_similarity": average_semantic_similarity,

        "average_generated_answer_length": mean(generated_lengths) if generated_lengths else 0.0,
        "average_reference_answer_length": mean(reference_lengths) if reference_lengths else 0.0,
    }

    return summary


def main():
    results = load_results()
    summary = calculate_summary_metrics(results)

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to: {METRICS_FILE}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()