"""
utils.py

Helper utilities for AI-based parameter selection in the
'do-android-wolves-dream-of-electric-sheep' project.
"""

import os
import openai
import re

# Optionally set your OPENAI_API_KEY here or in your environment
# os.environ["OPENAI_API_KEY"] = "sk-..."

# If you want to configure the organization or any other openai settings:
# openai.organization = "..."

def build_prompt(s, w, old_theta, step):
    """
    Build the prompt text to send to the LLM.

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

    Returns:
    --------
    prompt : str
        A text prompt describing the current system state and requesting a new theta.
    """
    prompt = (
        f"You are controlling the parameter 'theta' in a predator-prey model.\n\n"
        f"System state:\n"
        f"- Sheep (s): {s:.2f}\n"
        f"- Wolves (w): {w:.2f}\n"
        f"- Previous theta: {old_theta:.3f}\n"
        f"- Time step: {step}\n\n"
        "Goal:\n"
        "1. Prevent the wolves (w) from going extinct (i.e., w = 0).\n"
        "2. Maximize the wolf population.\n\n"
        "Please provide a new theta value in the range [0, 1], formatted as a decimal.\n"
        "No extra explanation is needed, just the number.\n"
    )
    return prompt

def call_llm(prompt, model="gpt-3.5-turbo", temperature=0.0, max_tokens=40):
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

def parse_theta_response(response, default=0.5):
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

def get_new_theta(s, w, old_theta, step, model="gpt-3.5-turbo"):
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
