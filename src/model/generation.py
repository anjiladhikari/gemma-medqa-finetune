from src.core.config import MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE,TOP_P,REPETITION_PENALTY


def build_messages(user_prompt: str):
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        }
    ]


def generate_response(pipe, user_prompt: str) -> str:
    messages = build_messages(user_prompt)

    output = pipe(
       messages,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    )

    return output[0]["generated_text"][-1]["content"]