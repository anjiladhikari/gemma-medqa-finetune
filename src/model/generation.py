"""
Handles the text generation logic.
This module is responsible for structuring prompts/messages and passing them 
to the Hugging Face pipeline to generate responses using predefined hyperparameters.
"""

from src.core.config import MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, TOP_P, REPETITION_PENALTY, SYSTEM_PROMPT


def build_messages(question: str):
    """
    Constructs a list of message dictionaries formatted for chat models.
    
    Args:
        question (str): The user's input/question.
        
    Returns:
        list: A dialogue history containing the system prompt and the user's question,
              formatted according to the expected input structure of conversational pipelines.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        },
    ]


def generate_response(pipe, user_prompt: str) -> str:
    """
    Generates a text response from the model using the provided Hugging Face pipeline.
    
    Args:
        pipe: The initialized Hugging Face text-generation pipeline.
        user_prompt (str): The specific question or prompt to generate a response for.
        
    Returns:
        str: The generated text response, stripped of leading/trailing whitespace.
    """
    # Transform the raw text prompt into the conversational message format
    messages = build_messages(user_prompt)

    # Pass the formatted messages along with strictly defined config parameters
    # to maintain consistency across generation attempts.
    output = pipe(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )

    # The pipeline returns a list of dictionaries. 
    # 'generated_text' contains the full dialogue history including the new generation appended at the end.
    # We retrieve the last item ([-1]) which is the assistant's generated reply, and clean up formatting.
    return output[0]["generated_text"][-1]["content"].strip()