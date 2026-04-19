"""
Benchmark execution script for the base generation model.
This script loads a small subset of the benchmarking dataset, runs text generation 
for each question, and saves the output alongside the reference answers for evaluation.
"""

import json

from src.core.config import TOP_K_QUESTIONS
from src.core.paths import BENCHMARK_FILE, BENCHMARK_OUTPUT_DIR, FIRST5_RESULTS_FILE, FULL_RESULTS_FILE
from src.model.load_base_model import load_text_generation_pipeline
from src.model.generation import generate_response


def load_benchmark_data() -> list:
    """
    Loads the benchmarking dataset from the configured JSON file.
    
    Returns:
        list: A list of dictionaries, where each dict represents a benchmark item 
              containing at least a 'question' and an 'answer'.
    """
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """
    Main entry point for running the base benchmark tests.
    Executes the text generation loop and saves the resulting comparisons.
    """
    # Ensure the output directory exists before making any file writes
    BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the full benchmark dataset and extract the top N requested items
    benchmark_data = load_benchmark_data()
    # first_five = benchmark_data[:TOP_K_QUESTIONS]
    if TOP_K_QUESTIONS is None:
        selected_data = benchmark_data
    else:
        selected_data = benchmark_data[:TOP_K_QUESTIONS]

    # Initialize the causal language model pipeline
    pipe = load_text_generation_pipeline()

    # Store results here to be written to disk at the end
    results = []

    # Iterate through the selected benchmark samples
    for idx, item in enumerate(selected_data, start=1):
        question = item["question"]
        reference_answer = item["answer"]

        # Provide a quick console logging footprint to track progress
        print(f"\nRunning question {idx}/{len(selected_data)}")
        print(f"Question: {question}")

        # Pass the extracted question to the model to generate a response
        generated_answer = generate_response(pipe, question)

        # Structure the combined item including the model's generated answer
        result_item = {
            "id": idx,
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
        }
        results.append(result_item)

    output_file = FULL_RESULTS_FILE if TOP_K_QUESTIONS is None else FIRST5_RESULTS_FILE

    # Save the aggregated output list into a formatted JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()