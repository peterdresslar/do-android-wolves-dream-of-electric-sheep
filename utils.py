"""
utils.py

Helper utilities for AI-based parameter selection in the
'do-android-wolves-dream-of-electric-sheep' project.

we may deal with multiple LLMs with different schemas.
"""

import os
from dotenv import load_dotenv
import openai
import re
from typing import Optional
from dataclasses import dataclass


MODEL = "gpt-4o-mini" # for now

# Load keys from .env file. See .env.local.example
load_dotenv('.env.local')

# If you want to configure the organization or any other openai settings:
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class WolfResponse:
    theta: float
    explanation: Optional[str] = None
    vocalization: Optional[str] = None

def build_prompt_high_information(
    s: float,
    w: float,
    old_theta: float,
    step: int,
    s_max: float,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    delta: Optional[float] = None,
    extra_hints: bool = True,
    respond_verbosely: bool = True
) -> str:
    """
    Build the prompt text to send to the LLM, providing more details
    about the predator-prey model (Lotka-Volterra) plus new concepts
    like 'crowding' or 'fighting' among wolves.

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
    extra_hints : bool
        If True, the LLM will be given extra hints about the model.
    give_explanation : bool
        If True, the LLM will provide an explanation of the wolf's reasoning.
    give_vocalization : bool
        If True, the LLM will provide a vocalization of the wolf's mood.

    Returns:
    --------
    prompt : str
        A text prompt that describes the system state, along with
        new conceptual details about crowding/fighting, requesting
        a new theta to balance classic LV behavior and new constraints.
    """
    prompt = [
        "You are controlling the parameter 'theta' in a modified Lotka-Volterra model.",
        "Wolves can exhibit 'competitive' or 'fighting' behavior, which adds new dynamics:",
        "",
        "Current system state:",
        f"- Time step: {step}",
        f"- Sheep (s): {s:.2f}",
        f"- Wolves (w): {w:.2f}",
        f"- Previous theta: {old_theta:.3f}",
        f"- Maximum sheep capacity (s_max): {s_max:.2f}",
    ]

    # Optionally show LV parameters
    if alpha is not None:
        prompt.append(f"- alpha (sheep growth rate): {alpha}")
    if beta is not None:
        prompt.append(f"- beta (predation rate): {beta}")
    if gamma is not None:
        prompt.append(f"- gamma (wolf death rate): {gamma}")
    if delta is not None:
        prompt.append(f"- delta (conversion rate): {delta}")

    prompt.append("")

    # Explanation of theta, crowding/fighting, and the model's objectives
    if extra_hints:
            prompt.append(
                "Parameter 'theta' controls both hunting intensity AND wolf population growth. "
                "Higher theta increases immediate hunting success but risks over-depleting sheep, "
                "while also accelerating wolf population growth (which could lead to future crowding). "
                "You must balance short-term gains with long-term sustainability. All wolves in the system "
                "use this same theta-selection strategy, so consider collective impacts."
            )
    else:
        prompt.append(
            "In this model, 'theta' adjusts how aggressively wolves hunt when sheep are scarce. "
            "We also consider potential crowding (intra-species competition) among wolves. "
            "Thus, theta, a float with up to 2 decimal places, ranges from 0.0 (very cautious) to 1.0 (fully aggressive)."
        )

    # Updated objectives
    prompt.append("Your main objectives:")
    prompt.append("1. Ensure wolf population survives indefinitely (never reach w=0)")
    prompt.append("2. Maintain sheep population above critical threshold as you determine.")
    prompt.append("3. Anticipate crowding effects from wolf population growth")
    prompt.append("4. Account for other wolves' identical decision-making process")
   

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
                "theta": 0.5,  // float between 0 and 1
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
    step: int,
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
        "You are the alpha wolf. Your choice affects both current hunting AND future pack growth.",
        "Other wolf packs use the same strategy - consider their likely choices when deciding.",
        "",
        f"Step {step} - Current situation:",
        f"- Sheep: {s:.2f} ({sheep_trend} by {abs(delta_s):.2f})",
        f"- Wolves: {w:.2f} ({wolves_trend} by {abs(delta_w):.2f})",
        f"- Last aggression: {old_aggression:.2f}",
        "",
        "Choose NEW aggressiveness (a float with up to 2 decimal places between 0 and 1):",
        "- High values: More food now, faster pack growth (risk future overcrowding)",
        "- Low values: Conserve sheep, slower growth (risk starvation)",
        "",
        "Remember: Other packs are making this same decision. Find balance between "
        "immediate needs and long-term survival."
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

def call_llm(
    prompt: str,
    model: str = MODEL,
    temperature: float = 0.0,
    max_tokens: int = 40
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
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You decide 'theta' in [0,1] for the predator-prey model."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]

def parse_wolf_response(
    response: str,
    default: float = 1.0
) -> WolfResponse:
    """
    Parse the LLM's response for a float value and clamp it to [0,1].

    Parameters:
    -----------
    response : str
        Raw text from the LLM.
    default : float
        Fallback if we can't parse a valid number.

    Returns:
    --------
    wolf_response : WolfResponse
        Includes the extracted theta value (0 <= theta <= 1),
        and optional explanation and vocalization.
    """
    # Find the first number in the response
    match = re.search(r"(\d*\.?\d+)", response)
    if match:
        try:
            theta_val = float(match.group(1))
        except ValueError:
            # If something strange came back
            theta_val = default
    else:
        # If no number is found
        theta_val = default

    # Clamp to [0,1]
    theta_val = max(0.0, min(1.0, theta_val))

    # Parse the explanation and vocalization
    explanation = None
    vocalization = None
    
    expl_match = re.search(r"Explanation: (.*)", response)
    if expl_match:
        explanation = expl_match.group(1)
    
    vocal_match = re.search(r"Vocalization: (.*)", response)
    if vocal_match:
        vocalization = vocal_match.group(1)

    return WolfResponse(
        theta=theta_val,
        explanation=explanation,
        vocalization=vocalization
    )

def get_new_theta(
    s: float,
    w: float,
    old_theta: float,
    step: int,
    model: str = MODEL,
    prompt_builder: callable = build_prompt_high_information
) -> float:
    """
    High-level function to build the prompt, call the LLM,
    parse the response, and return a new theta.

    Parameters:
    -----------
    s : float
        Current number of sheep.
    w : float
        Current number of wolves.
    old_theta : float
        The previous step's theta value.
    step : int
        Simulation time step.
    model : str
        Which LLM model to call.

    Returns:
    --------
    new_theta : float
        A valid theta value in [0,1].
    """
    prompt = prompt_builder(s, w, old_theta, step)
    response = call_llm(prompt, model=model)
    wolf_response = parse_wolf_response(response, default=old_theta)
    return wolf_response.theta
