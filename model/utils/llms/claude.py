# claude.py

import os

from anthropic import Anthropic

# Import shared data types
from model.utils.data_types import Usage, WolfResponse
from model.utils.init_utils import load_environment

# Load environment variables once
load_environment()

# Get API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set")

# Initialize client
client = Anthropic(api_key=api_key)


def call_claude(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    usage: Usage = None,
) -> str:
    """
    Call the Claude API with the given prompt.
    """
    # Get model from params if not explicitly provided
    if model is None:
        raise ValueError("Model is not provided")

    # Set defaults
    max_tokens = max_tokens if max_tokens is not None else 1024
    temperature = temperature if temperature is not None else 0.2

    # Call the API
    message = client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        temperature=temperature,
    )

    # Extract text content from the response
    # Claude 3 returns content as a list of content blocks
    response_text = ""
    if isinstance(message.content, list):
        for content_block in message.content:
            if content_block.type == "text":
                response_text += content_block.text
    else:
        # Fallback for older versions or if the API changes
        response_text = str(message.content)

    # Update usage if available (Claude doesn't provide token counts in the same way)
    if usage is not None:
        # Estimate tokens as a workaround
        prompt_tokens = len(prompt) // 4  # Rough estimate
        completion_tokens = len(response_text) // 4  # Rough estimate
        usage.add(prompt_tokens, completion_tokens, model)

    return response_text


async def call_claude_async(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    usage: Usage = None,
) -> str:
    """
    Async version of call_claude.
    Note: This is a placeholder - Anthropic's Python client doesn't have async support yet.
    """
    # For now, just call the synchronous version
    # In the future, this should be updated when Anthropic adds async support
    return call_claude(prompt, model, temperature, max_tokens, usage)


def get_claude_response(
    s: float,
    w: float,
    sheep_max: float,
    old_theta: float,
    step: int,
    respond_verbosely: bool = True,
    delta_s: float = 0,
    delta_w: float = 0,
    prompt_type: str = "high",
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    usage: Usage = None,
) -> WolfResponse:
    """
    Build a prompt, call Claude, parse the result into a WolfResponse.
    """
    # Import here to avoid circular imports
    from model.utils.llm_utils import (
        build_prompt_high_information,
        build_prompt_low_information,
        build_prompt_medium_information,
        parse_wolf_response,
    )

    # 1. Make the prompt based on prompt_type
    if prompt_type == "low":
        prompt = build_prompt_low_information(
            s=s,
            w=w,
            delta_s=delta_s,
            delta_w=delta_w,
            old_aggression=old_theta,
            respond_verbosely=respond_verbosely,
        )
    elif prompt_type == "medium":
        prompt = build_prompt_medium_information(
            s=s,
            w=w,
            old_theta=old_theta,
            step=step,
            sheep_max=sheep_max,
            respond_verbosely=respond_verbosely,
        )
    else:  # Default to high information
        prompt = build_prompt_high_information(
            s=s,
            w=w,
            old_theta=old_theta,
            step=step,
            sheep_max=sheep_max,
            respond_verbosely=respond_verbosely,
        )

    # 2. Get a raw string response from Claude
    response_str = call_claude(
        prompt, model=model, temperature=temperature, max_tokens=max_tokens, usage=usage
    )

    # 3. Parse that string into a WolfResponse
    wolf_resp = parse_wolf_response(response_str, prompt, default=old_theta)

    return wolf_resp


async def get_claude_response_async(
    s: float,
    w: float,
    sheep_max: float,
    old_theta: float,
    step: int,
    respond_verbosely: bool = True,
    delta_s: float = 0,
    delta_w: float = 0,
    prompt_type: str = "high",
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    usage: Usage = None,
) -> WolfResponse:
    """
    Async version of get_claude_response.
    """
    # Import here to avoid circular imports

    # For now, just call the synchronous version
    # In the future, this should be updated when Anthropic adds async support
    return get_claude_response(
        s=s,
        w=w,
        sheep_max=sheep_max,
        old_theta=old_theta,
        step=step,
        respond_verbosely=respond_verbosely,
        delta_s=delta_s,
        delta_w=delta_w,
        prompt_type=prompt_type,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        usage=usage,
    )
