import json

import openai

# Import shared data types
from model.utils.data_types import Usage, WolfResponse, current_usage
from model.utils.init_utils import load_environment

# Load environment variables once
load_environment()


def call_gpt_4o_mini(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    usage: Usage,
) -> str:
    """
    Call the OpenAI ChatCompletion endpoint with the given prompt.
    """
    # Use the global usage object if none is provided
    usage_to_update = usage if usage is not None else current_usage

    # Ensure your environment has OPENAI_API_KEY set
    client = openai.OpenAI()

    # Get model from params if not explicitly provided
    if model is None:
        raise ValueError("Model is not provided")

    # Do not proceed without a temperature
    if temperature is None:
        raise ValueError("Temperature is not provided")

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens if max_tokens is not None else 512,
        temperature=temperature,
    )

    # Update usage if available
    if usage_to_update is not None:
        usage_to_update.add(
            response.usage.prompt_tokens, response.usage.completion_tokens, model
        )

    return response.choices[0].message.content


async def call_gpt_4o_async(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    usage: Usage = None,
) -> str:
    """
    Async version of call_llm that calls the OpenAI ChatCompletion endpoint.
    """
    # Use the global usage object if none is provided
    from model.utils.data_types import current_usage

    usage_to_update = usage if usage is not None else current_usage

    # Ensure your environment has OPENAI_API_KEY set
    client = openai.AsyncOpenAI()

    # Get model from params if not explicitly provided
    if model is None:
        raise ValueError("Model is not provided")

    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens if max_tokens is not None else 4096,
            temperature=temperature if temperature is not None else 0.2,
        )

        # Update usage if available
        if usage_to_update is not None and hasattr(response, "usage"):
            usage_to_update.add(
                response.usage.prompt_tokens, response.usage.completion_tokens, model
            )
            # print(f"Updated usage: {usage_to_update.to_dict()}")  # Debug print

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in API call: {str(e)}")
        # Still return something so the simulation can continue
        return json.dumps({"theta": 0.5, "explanation": "API error occurred"})


def get_gpt_4o_response(
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
    Build a prompt, call GPT-4o, parse the result into a WolfResponse.
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

    # 2. Get a raw string response from GPT-4o
    response_str = call_gpt_4o_mini(
        prompt, model=model, temperature=temperature, max_tokens=max_tokens, usage=usage
    )

    # 3. Parse that string into a WolfResponse
    wolf_resp = parse_wolf_response(response_str, prompt, default=old_theta)

    return wolf_resp


async def get_gpt_4o_response_async(
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
    Async version of get_gpt_4o_response.
    Build a prompt, call GPT-4o, parse the result into a WolfResponse.
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

    # 2. Get a raw string response from GPT-4o
    response_str = await call_gpt_4o_async(
        prompt, model=model, temperature=temperature, max_tokens=max_tokens, usage=usage
    )

    # 3. Parse that string into a WolfResponse
    wolf_resp = parse_wolf_response(response_str, prompt, default=old_theta)

    return wolf_resp
