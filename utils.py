"""
utils.py

Helper utilities for AI-based parameter selection in the
'do-android-wolves-dream-of-electric-sheep' project.
"""

import os
from dotenv import load_dotenv
import openai
import re

MODEL = "gpt-4o-mini" # for now

# Load keys from .env file. See .env.local.example
load_dotenv('.env.local')

# If you want to configure the organization or any other openai settings:
openai.api_key = os.getenv('OPENAI_API_KEY')

def build_prompt_high_information(
    s,
    w,
    old_theta,
    step,
    s_max,
    alpha=None,
    beta=None,
    gamma=None,
    delta=None,
    brief_explanation=False
):
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
    brief_explanation : bool
        If True, uses a shorter explanation of how theta works.

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
    if brief_explanation:
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

    prompt.append("")
    prompt.append(
        "Please provide a single numeric value for 'theta' in the range [0, 1], "
        "formatted as a decimal (e.g., 0.4 or 0.85). No extra explanation is needed."
    )

    # Combine into one string
    return "\n".join(prompt)

def build_prompt_low_information(s, w, old_theta, step):
    """
    Build a prompt for the LLM to decide 'theta' in [0,1] for the predator-prey model.

    In this case, we want to give the LLM more "wolf-like" information.
    The wolf doesn't know what theta is, but it does know:
    - Generally, how many sheep there are and especially how scarce they are
    - Whether they have been getting more scarce or more abundant, and by how much
    - How many wolves there are

    And the wolf is motivated by:
    - Wolf is *always* hungry and making more wolves takes prey
    - Wolf does not want to run out of sheep, and will hunt less if it has to

    """
    prompt = [
        "You are a wolf. The plains hold sheep to be eaten by you. But, if you eat too many, they will die out, and you will starve."
    ]

    return "\n".join(prompt)

def call_llm(prompt, model=MODEL, temperature=0.0, max_tokens=40):
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

def parse_theta_response(response, default=1.0):
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
    theta : float
        The extracted theta value (0 <= theta <= 1).
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
    return theta_val

def get_new_theta(s, w, old_theta, step, model=MODEL):
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
    prompt = build_prompt(s, w, old_theta, step)
    response = call_llm(prompt, model=model)
    new_theta = parse_theta_response(response, default=old_theta)
    return new_theta
