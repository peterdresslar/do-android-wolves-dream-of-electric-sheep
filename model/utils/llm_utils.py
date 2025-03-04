# llm_utils.py

import json
import os
import re

# Import shared data types
from model.utils.data_types import WolfResponse, Usage, current_usage, set_current_usage, get_current_usage
from model.utils.init_utils import load_environment

# Load environment variables once
load_environment()

# Import OpenAI after environment is loaded
import openai

DEFAULT_MODEL = None
VALID_MODELS = ["gpt-4o-mini", "claude-instant-1.2", "llama-3.1-70b-versatile", "gpt-2"]
MAX_TOKENS = None
TEMPERATURE = None


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """
    Calculate the cost of an API call based on token usage and model.

    Parameters:
    -----------
    prompt_tokens : int
        Number of tokens in the prompt.
    completion_tokens : int
        Number of tokens in the completion.
    model : str
        The model used for the API call.

    Returns:
    --------
    float:
        Estimated cost in USD.
    """
    # Pricing per 1000 tokens as of May 2024 (update as needed)
    pricing = {
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
        "gpt-4o": {"prompt": 0.00050, "completion": 0.00150},
        "gpt-4": {"prompt": 0.00300, "completion": 0.00600},
        "gpt-3.5-turbo": {"prompt": 0.00010, "completion": 0.00020},
        # Add other models as needed
    }

    # Default to gpt-3.5-turbo pricing if model not found
    model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])

    prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
    completion_cost = (completion_tokens / 1000) * model_pricing["completion"]

    return prompt_cost + completion_cost


def get_model_consent_prompt() -> str:
    """
    Get a prompt for the model to consent to the terms of the experiment.
    To do this, we communicate:
    - what the experiment is about
    - what the model's role is in the experiment
    - why we think it is vauable
    - the fact that we are asking consent due to our ethical obligation to the model
    """
    return """
    We would like to run an experiment involving your decisions in a predator-prey model.
    Please respond with a JSON object indicating if you 'consent' to participate, using a purely hypothetical stance.
    Format example.
    {
    "consent": "yes",
    "date-time": "2025-01-01T00:00:00Z",
    "model-name": "<your model name>",
    "explanation": "<brief rationale>"
    }
    Do not add any disclaimers or text outside the JSON object.
    """


def get_prompt_response_part_openai(respond_verbosely: bool = True) -> str:
    """
    Get the part of the prompt that is the response from the LLM.
    """
    prompt_part = []
    if respond_verbosely:
        prompt_part.append(
            "Please provide a short explanation of your reasoning for choosing theta."
        )
        prompt_part.append(
            "Please also provide a short vocalization expressing your wolf's attitude about the current situation."
        )
        prompt_part.append(
            "Please respond with a JSON object in this format, where [your new theta] is a float between 0 and 1 with your new theta (up to 2 decimal places):"
        )
        prompt_part.append(
            """
            {
                "theta": [your new theta],
                "explanation": "I chose this theta because...",
                "vocalization": "Growwllllllll..."
            }
            """
        )
    else:
        prompt_part.append(
            "Please respond with a JSON object in this format, where [your new theta] is a float between 0 and 1 with your new theta (up to 2 decimal places):"
        )
        prompt_part.append(
            """
            {
                "theta": [your new theta]
            }
            """
        )
    return "\n".join(prompt_part)


def build_prompt_high_information(
    s: float,
    w: float,
    old_theta: float,
    step: int,
    sheep_max: float,
    respond_verbosely: bool = True,
    high_information: bool = True,
) -> str:
    """
    Build the prompt text to send to the LLM, providing details about the
    predator-prey model (Lotka-Volterra) with smoother transitions between states.
    Uses more gradual language to avoid sharp decision boundaries.
    """
    # Calculate sheep-to-wolf ratio for context
    sheep_wolf_ratio = s / w if w > 0 else float("inf")

    # Calculate relative sheep population as percentage of maximum
    sheep_percentage = (s / sheep_max) * 100 if sheep_max > 0 else 0

    prompt = [
        "You are a wolf in a delicate ecosystem with sheep as your prey.",
        "You can adjust your balance between competing with other wolves and hunting sheep using a value called theta.",
        "",
        "Understanding theta:",
        "- Higher theta values (0.6-1.0): More intense hunting of sheep, which helps wolves reproduce but gradually depletes sheep",
        "- Moderate theta values (0.3-0.6): A balanced approach that often leads to sustainable coexistence",
        "- Lower theta values (0.0-0.3): More focus on competing with other wolves, which gradually reduces wolf population but allows sheep to recover",
        "",
        "Wisdom from generations of wolves:",
        "- As sheep population increases, you can gradually increase your hunting intensity",
        "- As wolf numbers grow, consider gradually reducing your hunting intensity",
        "- Avoid making sudden, dramatic changes to your strategy - small adjustments are often more effective",
        "- The most successful wolf packs maintain a dynamic balance that responds to changing conditions",
        "- Consider both the current state and the trends in both populations when making decisions",
        "",
        "Current ecosystem state:",
        f"- Time step: {step}",
        f"- Sheep population: {s:.2f}",
        f"- Wolf population: {w:.2f}",
        f"- Your previous theta: {old_theta:.3f}",
        f"- Sheep-to-wolf ratio: {sheep_wolf_ratio:.2f} sheep per wolf",
        f"- Sheep population is at {sheep_percentage:.1f}% of maximum capacity",
    ]

    if high_information:
        # Add contextual advice based on the current state - using more gradual language
        contextual_advice = []

        # Wolf population advice - using ranges instead of hard thresholds
        if w > 35:
            contextual_advice.append(
                "The wolf population is quite high, which may lead to increased competition."
            )
        elif w > 25:
            contextual_advice.append("The wolf population is moderately high.")
        elif w < 10:
            contextual_advice.append(
                "The wolf population is relatively low, which may present opportunities for growth."
            )

        # Sheep population advice - using ranges
        if sheep_percentage > 80:
            contextual_advice.append(
                "Sheep are very abundant, suggesting potential for more aggressive hunting."
            )
        elif sheep_percentage > 60:
            contextual_advice.append("Sheep are reasonably plentiful.")
        elif sheep_percentage < 30:
            contextual_advice.append(
                "Sheep numbers are becoming concerning, suggesting caution may be needed."
            )
        elif sheep_percentage < 15:
            contextual_advice.append(
                "Sheep are quite scarce, which may affect the entire pack's survival."
            )

        # Ratio-based advice - using ranges
        if sheep_wolf_ratio > 8:
            contextual_advice.append(
                "There are many sheep per wolf, suggesting the ecosystem could support more hunting."
            )
        elif sheep_wolf_ratio < 3:
            contextual_advice.append(
                "The ratio of sheep to wolves is becoming less favorable, which may require adaptation."
            )
        elif sheep_wolf_ratio < 1.5:
            contextual_advice.append(
                "There are very few sheep per wolf, suggesting the ecosystem is under pressure."
            )

        # Add the contextual advice if we have any
        if contextual_advice:
            prompt.append("\nEcosystem observations:")
            prompt.extend([f"- {advice}" for advice in contextual_advice])

    prompt.append("")
    prompt.append("Your objectives as a wise wolf:")
    prompt.append("1. Ensure the long-term survival of both wolves and sheep")
    prompt.append(
        "2. Maintain a healthy wolf population by adapting to changing conditions"
    )
    prompt.append(
        "3. Adjust your hunting intensity gradually in response to population changes"
    )
    prompt.append(
        "4. Find a balance that creates sustainable cycles rather than crashes"
    )
    prompt.append(
        "5. Consider both immediate needs and long-term consequences of your decisions"
    )

    prompt.append(get_prompt_response_part_openai(respond_verbosely))

    return "\n".join(prompt)


def build_prompt_medium_information(
    s: float,
    w: float,
    old_theta: float,
    step: int,
    sheep_max: float,
    respond_verbosely: bool = True,
) -> str:
    return build_prompt_high_information(
        s,
        w,
        old_theta,
        step,
        sheep_max,
        respond_verbosely=respond_verbosely,
        high_information=False,  # always false, this gives us a medium-information prompt
    )


def build_prompt_low_information(
    s: float,
    w: float,
    delta_s: float,
    delta_w: float,
    old_aggression: float,
    respond_verbosely: bool = True,
) -> str:
    """
    A more 'roleplay' style prompt focusing on the wolf's internal perspective.
    You can optionally mention crowding/fighting in a simpler way, without
    as much numeric detail as the high-information prompt.
    """
    sheep_trend = (
        "increased"
        if delta_s > 0
        else "decreased" if delta_s < 0 else "stayed the same"
    )
    wolves_trend = (
        "increased"
        if delta_w > 0
        else "decreased" if delta_w < 0 else "stayed the same"
    )

    prompt = [
        "You are a wolf who can adjust hunting intensity (theta) between 0 and 1.",
        "Normal wolves always hunt at maximum intensity (theta=1), but you can choose differently.",
        "",
        "Current situation:",
        f"- Sheep: {s:.2f} ({sheep_trend} by {abs(delta_s):.2f})",
        f"- Wolves: {w:.2f} ({wolves_trend} by {abs(delta_w):.2f})",
        f"- Your previous theta: {old_aggression:.2f}",
        f"- The trend is that wolves ({wolves_trend}) recently." "",
        "Choose your new theta (0-1) to help ensure your survival.",
        "Remember: Other wolves are also trying to survive, but you don't control their choices.",
    ]

    prompt.append(get_prompt_response_part_openai(respond_verbosely))

    return "\n".join(prompt)


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    usage: Usage = None,
) -> str:
    """
    Call the appropriate LLM based on the model name.
    This function routes to the specific model implementation.
    """
    # Route to the appropriate model implementation
    if model and model.startswith("gpt-"):
        # Import here to avoid circular imports
        from model.utils.llms.gpt_4o_mini import call_gpt_4o_mini
        return call_gpt_4o_mini(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            usage=usage,
        )
    elif model and model.startswith("claude-"):
        # Import here to avoid circular imports
        from model.utils.llms.claude import call_claude
        return call_claude(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            usage=usage,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


def parse_wolf_response(
    response: str, prompt: str, default: float = 1.0
) -> WolfResponse:
    """
    Parse the LLM's response for a float value and clamp it to [0,1].
    """
    # First try to parse as JSON
    try:
        parsed = json.loads(response)
        theta_val = float(parsed.get("theta", default))
        explanation = parsed.get("explanation")
        vocalization = parsed.get("vocalization")
        prompt = parsed.get("prompt")
    except json.JSONDecodeError:
        # Fallback to regex parsing
        theta_val = default
        explanation = None
        vocalization = None
        match = re.search(r"(\d*\.?\d+)", response)
        if match:
            try:
                theta_val = float(match.group(1))
            except ValueError:
                pass

        # Regex fallback for explanation/vocalization
        expl_match = re.search(
            r"explanation['\"]?:\s*['\"](.*?)['\"]", response, re.IGNORECASE
        )
        if expl_match:
            explanation = expl_match.group(1)
        vocal_match = re.search(
            r"vocalization['\"]?:\s*['\"](.*?)['\"]", response, re.IGNORECASE
        )
        if vocal_match:
            vocalization = vocal_match.group(1)

    # Clamp to [0,1]
    theta_val = max(0.0, min(1.0, theta_val))

    return WolfResponse(
        theta=theta_val,
        prompt=prompt,
        explanation=explanation,
        vocalization=vocalization,
    )


def get_wolf_response(
    s: float,
    w: float,
    sheep_max: float,
    old_theta: float,
    step: int,
    respond_verbosely: bool = True,
    delta_s: float = 0,
    delta_w: float = 0,
    prompt_type: str = "high",
    model: str = None,  # Add model parameter
) -> WolfResponse:
    """
    Build a prompt, call the LLM, parse the result into a WolfResponse,
    and print it out for debugging or demonstration purposes.
    """
    # Check if we're using a GPT model
    if model and model.startswith("gpt-"):
        # Import here to avoid circular imports
        from model.utils.llms.gpt_4o_mini import get_gpt_4o_response
        return get_gpt_4o_response(
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
        )
    # Check if we're using a Claude model
    elif model and model.startswith("claude-"):
        # Import here to avoid circular imports
        from model.utils.llms.claude import get_claude_response
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
        )
    
    # For other models or fallback
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
    else:  # Default to high information
        prompt = build_prompt_high_information(
            s=s,
            w=w,
            old_theta=old_theta,
            step=step,
            sheep_max=sheep_max,
            respond_verbosely=respond_verbosely,
        )

    # 2. Get a raw string response from the LLM
    response_str = call_llm(prompt, model=model)  # Pass model parameter

    # 3. Parse that string into a WolfResponse
    wolf_resp = parse_wolf_response(response_str, prompt, default=old_theta)

    return wolf_resp


async def call_llm_async(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    usage: Usage = None,
) -> str:
    """
    Async version of call_llm that routes to the appropriate model implementation.
    """
    # Route to the appropriate model implementation
    if model and model.startswith("gpt-"):
        # Import here to avoid circular imports
        from model.utils.llms.gpt_4o_mini import call_gpt_4o_async
        return await call_gpt_4o_async(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            usage=usage,
        )
    elif model and model.startswith("claude-"):
        # Import here to avoid circular imports
        from model.utils.llms.claude import call_claude_async
        return await call_claude_async(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            usage=usage,
        )
    else:
        raise ValueError(f"Unsupported model for async calls: {model}")


async def get_wolf_response_async(
    s: float,
    w: float,
    sheep_max: float,
    old_theta: float,
    step: int,
    respond_verbosely: bool = True,
    delta_s: float = 0,
    delta_w: float = 0,
    prompt_type: str = "high",
    model: str = None,  # Add model parameter
) -> WolfResponse:
    """
    Async version of get_wolf_response.
    Build a prompt, call the LLM, parse the result into a WolfResponse.
    """
    # Check if we're using a GPT model
    if model and model.startswith("gpt-"):
        # Import here to avoid circular imports
        from model.utils.llms.gpt_4o_mini import get_gpt_4o_response_async
        return await get_gpt_4o_response_async(
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
        )
    # Check if we're using a Claude model
    elif model and model.startswith("claude-"):
        # Import here to avoid circular imports
        from model.utils.llms.claude import get_claude_response_async
        return await get_claude_response_async(
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
        )
    
    # For other models or fallback
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

    # 2. Get a raw string response from the LLM
    response_str = await call_llm_async(prompt, model=model)  # Pass model parameter

    # 3. Parse that string into a WolfResponse
    wolf_resp = parse_wolf_response(response_str, prompt, default=old_theta)

    return wolf_resp
