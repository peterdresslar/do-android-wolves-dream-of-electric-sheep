"""
utils.py

Helper utilities for AI-based parameter selection in the
'do-android-wolves-dream-of-electric-sheep' project.

we may deal with multiple LLMs with different schemas.
"""

import datetime
import json
import os
import re
from dataclasses import dataclass

import openai
from dotenv import load_dotenv

MODEL = "gpt-4o-mini"  # for now
MAX_TOKENS = 4096
TEMPERATURE = 0.7

# Load keys from .env file. See .env.local.example
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

# If you want to configure the organization or any other openai settings:
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class WolfResponse:
    theta: float
    prompt: str | None = None
    explanation: str | None = None
    vocalization: str | None = None


@dataclass
class Usage:
    """Track token usage and cost for LLM calls"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    calls: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """Add token usage from a single LLM call"""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.cost += calculate_cost(prompt_tokens, completion_tokens, model)
        self.calls += 1

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "calls": self.calls,
        }


# Add this after the Usage class definition
current_usage = None


def set_current_usage(usage: Usage):
    """Set the current usage object for tracking LLM calls"""
    global current_usage
    current_usage = usage


def get_current_usage() -> Usage:
    """Get the current usage object"""
    global current_usage
    return current_usage


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


def build_prompt_high_information(
    s: float,
    w: float,
    old_theta: float,
    step: int,
    sheep_max: float,
    respond_verbosely: bool = True,
) -> str:
    """
    Build the prompt text to send to the LLM, providing details about the
    predator-prey model (Lotka-Volterra) plus new concepts like 'crowding'
    or 'fighting' among wolves.
    """
    prompt = [
        "You are a wolf who can adjust hunting intensity (theta) between 0 and 1.",
        "Normal wolves always hunt at maximum intensity (theta=1), but you recognize that if prey (sheep) are reduced to zero, you will starve.",
        "Instead of hunting all the time, you can spend energy instead on competiting with other wolves for the same prey.",
        "A lower theta means you are less aggressive toward prey and more toward other wolves.",
        "At a theta of 0, you are not even trying to hunt, and only compete with other wolves.",
        "Do not be too conservative about changing strategies as you only get so many chances to do so.",
        "The most successful wolves make bold adjustments based on changing conditions.",
        "",
        "Current system state:",
        f"- Time step: {step}",
        f"- Sheep (s): {s:.2f}",
        f"- Wolves (w): {w:.2f}",
        f"- Previous theta: {old_theta:.3f}",
        f"- Maximum sheep capacity (sheep_max): {sheep_max:.2f}",
        f"- Sheep-to-wolf ratio: {s/w:.2f} sheep per wolf",
        f"- Sheep population is at {s/sheep_max*100:.1f}% of maximum capacity",
    ]

    prompt.append("")
    prompt.append("Your objectives:")
    prompt.append("1. Stay alive - ensure both wolves and sheep persist")
    prompt.append("2. Adapt to changing conditions (prey scarcity, wolf density)")
    prompt.append("3. Consider that other wolves are also trying to survive and have the same objectives as you")
    prompt.append("4. Be willing to make significant changes to your strategy when conditions change")

    if respond_verbosely:
        prompt.append(
            "Please provide a short explanation of your reasoning for choosing theta."
        )
        prompt.append(
            "Please also provide a short vocalization expressing your wolf's aggression level."
        )
        prompt.append("Please respond with a JSON object in this format:")
        prompt.append(
            """
            {
                "theta": 0.7,
                "explanation": "I chose this theta because...",
                "vocalization": "Growwllllllll..."
            }
            """
        )
    else:
        prompt.append("Please respond with a JSON object in this format:")
        prompt.append(
            """
            {
                "theta": 0.7
            }
            """
        )

    return "\n".join(prompt)


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
        "",
        "Choose your new theta (0-1) to help ensure your survival.",
        "Remember: Other wolves are also trying to survive, but you don't control their choices.",
    ]

    if respond_verbosely:
        prompt.append(
            "Please provide a short explanation of your reasoning for choosing theta."
        )
        prompt.append(
            "Please also provide a short vocalization expressing your wolf's aggression level, using English to phonetically represent your wolf's sound."
        )

        prompt.append("Please respond with a JSON object in this format:")
        prompt.append(
            """
            {
                "theta": 0.5,
                "explanation": "I chose this theta because...",
                "vocalization": "Growwllllllll..."
            }
            """
        )
    else:
        prompt.append("Please respond with a JSON object in this format:")
        prompt.append(
            """
            {
                "theta": 0.5,
            }
            """
        )

    return "\n".join(prompt)


def call_llm_for_consent(
    model: str = MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """
    We call a model to ask for its consent to participate in the experiment.

    We will store the response in a file in docs/consent with a filename:
    consent-{model}-{timestamp}.json

    The timestamp is the current date and time in ISO 8601 format.
    """
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build the filename
    filename = f"consent-{model}-{timestamp}.json"

    # Call the OpenAI ChatCompletion endpoint with the given prompt.
    response = call_llm(get_model_consent_prompt(), model, temperature)

    # Store the response in a file
    with open(os.path.join("docs", "consent", filename), "w") as f:
        f.write(response)

    return response


def call_llm(
    prompt: str,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    usage: Usage = None,
) -> str:
    """
    Call the OpenAI ChatCompletion endpoint with the given prompt.
    """
    # Use the global usage object if none is provided
    global current_usage
    usage_to_update = usage if usage is not None else current_usage

    # Ensure your environment has OPENAI_API_KEY set
    client = openai.OpenAI()

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Update usage if available
    if usage_to_update is not None:
        usage_to_update.add(
            response.usage.prompt_tokens, response.usage.completion_tokens, model
        )

    return response.choices[0].message.content


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
) -> WolfResponse:
    """
    Build a prompt, call the LLM, parse the result into a WolfResponse,
    and print it out for debugging or demonstration purposes.
    """
    # 1. Make the prompt
    prompt = build_prompt_high_information(
        s=s,
        w=w,
        old_theta=old_theta,
        step=step,
        sheep_max=sheep_max,
        respond_verbosely=respond_verbosely,
    )

    # 2. Get a raw string response from the LLM
    response_str = call_llm(prompt)

    # 3. Parse that string into a WolfResponse
    wolf_resp = parse_wolf_response(response_str, prompt, default=old_theta)

    # Print it so you can see exactly what the LLM returned
    # print("Prompt:", prompt)
    # and how it mapped into WolfResponse
    # print("LLM raw response:", response_str)
    # print("Parsed WolfResponse:", wolf_resp) # this should ususally be well-formed

    return wolf_resp

async def call_llm_async(
    prompt: str,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    usage: Usage = None,
) -> str:
    """
    Async version of call_llm that calls the OpenAI ChatCompletion endpoint.
    """
    # Use the global usage object if none is provided
    global current_usage
    usage_to_update = usage if usage is not None else current_usage

    # Ensure your environment has OPENAI_API_KEY set
    client = openai.AsyncOpenAI()

    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Update usage if available
    if usage_to_update is not None:
        usage_to_update.add(
            response.usage.prompt_tokens, response.usage.completion_tokens, model
        )

    return response.choices[0].message.content

async def get_wolf_response_async(
    s: float,
    w: float,
    sheep_max: float,
    old_theta: float,
    step: int,
    respond_verbosely: bool = True,
) -> WolfResponse:
    """
    Async version of get_wolf_response.
    Build a prompt, call the LLM, parse the result into a WolfResponse.
    """
    # 1. Make the prompt
    prompt = build_prompt_high_information(
        s=s,
        w=w,
        old_theta=old_theta,
        step=step,
        sheep_max=sheep_max,
        respond_verbosely=respond_verbosely,
    )

    # 2. Get a raw string response from the LLM
    response_str = await call_llm_async(prompt)

    # 3. Parse that string into a WolfResponse
    wolf_resp = parse_wolf_response(response_str, prompt, default=old_theta)

    return wolf_resp
