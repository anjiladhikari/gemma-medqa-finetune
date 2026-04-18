from pathlib import Path

from src.model.load_base_model import load_text_generation_pipeline
from src.model.generation import generate_response


PROMPTS = [
    "Explain overfitting in simple words.",
    "Which medicine shall i take when i get fever?",
    "What is panadol?",
]


def main():
    pipe = load_text_generation_pipeline()

    output_dir = Path("outputs/exploration")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"\n--- Prompt {i} ---")
        print("PROMPT:")
        print(prompt)

        response = generate_response(pipe, prompt)

        print("\nRESPONSE:")
        print(response)
        print("-" * 60)

        out_file = output_dir / f"prompt_{i}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("PROMPT:\n")
            f.write(prompt + "\n\n")
            f.write("RESPONSE:\n")
            f.write(response)


if __name__ == "__main__":
    main()