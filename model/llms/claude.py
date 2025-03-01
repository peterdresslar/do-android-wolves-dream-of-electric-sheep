# claude.py

import os

from anthropic import Anthropic
from dotenv import load_dotenv

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set")

client = Anthropic(
    api_key=api_key,  # This is the default and can be omitted
)





def call_llm(model: str, prompt: str) -> str:
    """
    Call the LLM with the given prompt.
    """
    message = client.messages.create(
        max_tokens=1024,
        messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
        model=model,
    )
    return message.content
