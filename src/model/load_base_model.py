import torch
from transformers import pipeline

from src.core.config import MODEL_ID


def load_text_generation_pipeline():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        task="text-generation",
        model=MODEL_ID,
        device_map="auto",
        torch_dtype=dtype,
    )
    return pipe