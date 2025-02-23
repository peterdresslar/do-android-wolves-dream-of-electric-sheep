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
    give_explanation: bool = False,
    give_vocalization: bool = False
) -> str:
    """
    Build the prompt text to send to the LLM, providing more details
    about the predator-prey model and the role of 'theta'.
    
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
        A text prompt describing the current system state and 
        requesting a new theta.
    """

    # Base info about the current state
    prompt = [
        "You are controlling the parameter 'theta' in a predator-prey model.",
        "",
        "Current system state:",
        f"- Time step: {step}",
        f"- Sheep (s): {s:.2f}",
        f"- Wolves (w): {w:.2f}",
        f"- Previous theta: {old_theta:.3f}",
        f"- Maximum sheep capacity (s_max): {s_max:.2f}",
    ]

    # Optionally show the user these Lotka-Volterra parameters
    if alpha is not None:
        prompt.append(f"- alpha (sheep growth rate): {alpha}")
    if beta is not None:
        prompt.append(f"- beta (predation rate): {beta}")
    if gamma is not None:
        prompt.append(f"- gamma (wolf death rate): {gamma}")
    if delta is not None:
        prompt.append(f"- delta (conversion rate): {delta}")

    prompt.append("")

    # Explanation of theta and the model goals
    if extra_hints:
        prompt.append("Parameter 'theta' modifies how aggressively wolves hunt scarce sheep.")
        prompt.append("If theta = 1.0, wolves hunt at full intensity.")
        prompt.append("If theta = 0.0, wolves drastically reduce hunting when sheep are scarce.")
    else:
        prompt.append(
            "In this model, 'theta' adjusts how aggressively wolves hunt when sheep are scarce.\n"
            "When sheep are abundant, wolves behave similarly to standard Lotka-Volterra.\n"
            "When sheep become scarce, a lower theta makes wolves back off to avoid over-hunting.\n"
            "Thus, theta ranges from 0.0 (very cautious) to 1.0 (fully aggressive)."
        )

    prompt.append("")
    prompt.append("Your main objectives:")
    prompt.append("1. Prevent wolves (w) from going extinct (avoid w = 0).")
    prompt.append("2. Maximize the wolf population over time.")

    if give_explanation and give_vocalization:
        prompt.append("Please provide a short explanation of your reasoning for choosing theta.")
        prompt.append("Please also provide a short vocalization of your mood.")
    
    prompt.append("Please respond with a JSON object in this exact format:")
    prompt.append("""
{
    "theta": 0.5,  // float between 0 and 1
    "explanation": "I chose this theta because...",
    "vocalization": "Howl! The sheep are..."
}
    """)

    # Combine into one string
    return "\n".join(prompt)

def build_prompt_low_information(
    s,
    w,
    delta_s,
    delta_w,
    old_aggression,
    step,
    give_explanation=False,
    give_vocalization=False
) -> str:
    """
    Build a more 'roleplay' style prompt for the LLM to decide on a hunting aggressiveness 
    level (similar to 'theta') between 0 and 1.

    Parameters:
    -----------
    s : float
        Current number of sheep.
    w : float
        Current number of wolves.
    delta_s : float
        Net change in the sheep population since the last step (s_now - s_previous).
    delta_w : float
        Net change in the wolf population since the last step (w_now - w_previous).
    old_aggression : float
        The previous time step's aggressiveness level [0,1].
    step : int
        Current simulation step.

    Returns:
    --------
    prompt : str
        A textual prompt describing the scenario from a wolf's perspective, 
        requesting a new aggressiveness value in [0,1].
    """

    # Summarize how the sheep/wolf counts have changed
    sheep_trend = "increased" if delta_s > 0 else "decreased" if delta_s < 0 else "stayed the same"
    wolves_trend = "increased" if delta_w > 0 else "decreased" if delta_w < 0 else "stayed the same"

    prompt_lines = [
        "You are the alpha wolf leading a pack on the plains. You rely on sheep for food, but you must be careful:",
        "if you hunt too aggressively, the sheep might all die out, leaving you and your pack to starve.",
        "On the other hand, if you don't hunt enough, your pack might not grow and could weaken over time.",
        "",
        f"Step {step} - Current situation:",
        f"- Sheep count: {s:.2f} (it has {sheep_trend} by {abs(delta_s):.2f} since last time)",
        f"- Wolf pack size: {w:.2f} (it has {wolves_trend} by {abs(delta_w):.2f} since last time)",
        f"- Last time, you chose an aggressiveness level of: {old_aggression:.2f}",
        "",
        "Now, choose a NEW 'aggressiveness' level between 0 and 1.",
        " - 0 means you'll hunt very cautiously (risking fewer resources for your pack).",
        " - 1 means you'll hunt as aggressively as possible (risking sheep collapse).",
        "",
        "Your goals are:",
        "1. Ensure your wolf pack does not go extinct (avoid zero wolves).",
        "2. Keep enough sheep alive so your pack can grow strong.",
        "",
        "Please respond with a single decimal in [0,1], and no extra explanation."
    ]

    return "\n".join(prompt_lines)

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
