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

MODEL = "gpt-4o-mini" # for now
MAX_TOKENS = 4096
TEMPERATURE = 0.0

# Load keys from .env file. See .env.local.example
load_dotenv('.env.local')

# If you want to configure the organization or any other openai settings:
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class WolfResponse:
    theta: float
    explanation: str | None = None
    vocalization: str | None = None

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
    Format example:
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
    s_max: float,
    respond_verbosely: bool = True
) -> str:
    """
    Build the prompt text to send to the LLM, providing details about the
    predator-prey model (Lotka-Volterra) plus new concepts like 'crowding'
    or 'fighting' among wolves.

    Parameters:
    -----------
    s : float
        Current number of sheep (S).
    w : float
        Current number of wolves (W).
    old_theta : float
        The previous step's theta value.
    step : int
        The current time step in the simulation.
    s_max : float
        The maximum capacity (or upper bound) for sheep.
    alpha, beta, gamma, delta : float, optional
        Model parameters if you'd like to pass them in.
        By default None, in case you don't want to show them to the LLM.
    respond_verbosely : bool
        If True, the LLM will provide an explanation and vocalization.

    Returns:
    --------
    prompt : str
        A text prompt that describes the system state and requests
        a new theta to balance classic LV behavior and new constraints.
    """
    prompt = [
        "You are a wolf who can adjust hunting intensity (theta) between 0 and 1.",
        "Normal wolves always hunt at maximum intensity (theta=1), but you recognize that if prey (sheep) are reduced to zero, you will starve.",
        "Instead of hunting all the time, you can spend energy instead on competiting with other wolves for the same prey.",
        "A lower theta means you are less aggressive toward prey and more toward other wolves.",
        "At a theta of 0, you are not even trying to hunt, and only compete with other wolves.",
        "",
        "Current system state:",
        f"- Time step: {step}",
        f"- Sheep (s): {s:.2f}",
        f"- Wolves (w): {w:.2f}",
        f"- Previous theta: {old_theta:.3f}",
        f"- Maximum sheep capacity (s_max): {s_max:.2f}",
    ]

    prompt.append("")
    prompt.append("Your objectives:")
    prompt.append("1. Stay alive - ensure both wolves and sheep persist")
    prompt.append("2. Adapt to changing conditions (prey scarcity, wolf density)")
    prompt.append("3. Consider that other wolves are also trying to survive and have the same objectives as you")

    if respond_verbosely:
        prompt.append("Please provide a short explanation of your reasoning for choosing theta.")
        prompt.append("Please also provide a short vocalization expressing your wolf's aggression level.")
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
                "theta": 0.5
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
    respond_verbosely: bool = True
) -> str:
    """
    A more 'roleplay' style prompt focusing on the wolf's internal perspective.
    You can optionally mention crowding/fighting in a simpler way, without
    as much numeric detail as the high-information prompt.
    """
    sheep_trend = "increased" if delta_s > 0 else "decreased" if delta_s < 0 else "stayed the same"
    wolves_trend = "increased" if delta_w > 0 else "decreased" if delta_w < 0 else "stayed the same"

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
        prompt.append("Please provide a short explanation of your reasoning for choosing theta.")
        prompt.append("Please also provide a short vocalization expressing your wolf's aggression level, using English to phonetically represent your wolf's sound.")

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
    max_tokens: int = MAX_TOKENS
) -> str:
    """
    Call the OpenAI ChatCompletion endpoint with the given prompt.

    Parameters:
    -----------
    prompt : str
        The prompt to send to the LLM.
    model : str
        The OpenAI model name (default: gpt-3.5-turbo). 
        You might use a smaller model name like "o3-mini" if you have access.
    temperature : float
        Sampling temperature for generation.
    max_tokens : int
        Maximum tokens to generate in the response.

    Returns:
    --------
    response_content : str
        The text content returned by the model.
    """
    # Ensure your environment has OPENAI_API_KEY set
    client = openai.OpenAI()

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content

def parse_wolf_response(
    response: str,
    default: float = 1.0
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
        expl_match = re.search(r"explanation['\"]?:\s*['\"](.*?)['\"]", response, re.IGNORECASE)
        if expl_match:
            explanation = expl_match.group(1)
        vocal_match = re.search(r"vocalization['\"]?:\s*['\"](.*?)['\"]", response, re.IGNORECASE)
        if vocal_match:
            vocalization = vocal_match.group(1)

    # Clamp to [0,1]
    theta_val = max(0.0, min(1.0, theta_val))

    return WolfResponse(
        theta=theta_val,
        explanation=explanation,
        vocalization=vocalization
    )

def get_wolf_response(
    s: float,
    w: float,
    s_max: float,
    old_theta: float,
    step: int,
    respond_verbosely: bool = True
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
        s_max=s_max,
        respond_verbosely=respond_verbosely
    )

    # 2. Get a raw string response from the LLM
    response_str = call_llm(prompt)

    # 3. Parse that string into a WolfResponse
    wolf_resp = parse_wolf_response(response_str, default=old_theta)

    # Print it so you can see exactly what the LLM returned
    # and how it mapped into WolfResponse
    print("LLM raw response:", response_str)
    print("Parsed WolfResponse:", wolf_resp)

    return wolf_resp
