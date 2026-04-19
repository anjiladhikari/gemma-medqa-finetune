"""
Configuration file for the Gemma MedQA finetuning project.
Contains model and generation hyperparameters.
"""

# The base model identifier from the Hugging Face Hub
MODEL_ID = "google/gemma-3-1b-it"

# The system prompt that determines the persona and constraint behavior of the LLM
SYSTEM_PROMPT ="""
You are a medical question-answering assistant.
Answer clearly, accurately, and concisely.in plain paragraph form.
Do not use bullet points, markdown, headings, bold text, or numbered lists.
Do not add disclaimers or extra warnings unless the question is specifically about emergency symptoms or urgent care.
Keep the answer concise and directly focused on the question.
Do not add unnecessary details.
"""

# --- Generation Hyperparameters ---

# The maximum number of tokens the model is allowed to output per generation
MAX_NEW_TOKENS = 250

# Controls randomness (lower = more deterministic, higher = more creative)
# Note: Takes effect only when DO_SAMPLE is True
TEMPERATURE = 0.2

# If False, uses Greedy decoding (always picks the most likely next token).
# If True, randomly samples tokens based on probabilities.
DO_SAMPLE = False

# The number of alternative sequences or items to operate on when relevant
TOP_K_QUESTIONS = None

# Nucleus sampling parameter: limits token selection to a subset whose cumulative probability is >= TOP_P.
# Note: Takes effect only when DO_SAMPLE is True
TOP_P = 0.9

# Penalty applied to prevent the model from repeating itself. 
# 1.0 means no penalty, values > 1.0 penalize repetition.
REPETITION_PENALTY = 1.1