# simulation_utils.py
"""
Mostly reference ODE functions
"""

import datetime
import json
import os

import numpy as np
import pandas as pd
from scipy.integrate import odeint

DEFAULT_RESULTS_PATH = "../data/results"

#################################################################
# BASIC STUFF
#################################################################
def round4(x) -> float:
    """
    Round a float to 4 decimal places.

    Use in CSV

    >>> round4(1.23456789)
    1.2346
    """
    return round(x, 4)

def format4(x) -> str:
    """
    Return a string from a float, rounded to four places and including trailing zeros.

    Use in JSON

    >>> format4(1.23456789)
    "1.2346"

    >>> format4(1.23)
    "1.2300"
    """
    return f"{x:.4f}"

#################################################################
# Reference ODE Functions
#################################################################
def dx_dt(x, t, alpha, beta, gamma, delta):
    s, w = x
    ds_dt = alpha * s - beta * s * w
    dw_dt = -gamma * w + delta * beta * s * w
    return [ds_dt, dw_dt]

def get_reference_ODE(model_params, model_time):
    alpha = model_params['alpha']
    beta = model_params['beta']
    gamma = model_params['gamma']
    delta = model_params['delta']

    t_end = model_time['time']
    times = np.linspace(0, t_end, model_time['tmax'])
    x0 = [model_params['s_start'], model_params['w_start']]

    integration = odeint(dx_dt, x0, times, args=(alpha, beta, gamma, delta)) # via cursor, verify this
    ode_df = pd.DataFrame({
        't': times,
        's': round4(integration[:,0]),
        'w': round4(integration[:,1])
    })
    return ode_df

#################################################################
# Simulation Telemetry
#################################################################
def save_simulation_results(results, results_path=None):
    """Save simulation parameters (starting conditions) and results information to a summary file and a directory with detailed results.
    The summary file should be named using base params first amd then a datetime timestamp
    For instance:
    [ModelName]_[Steps]_[S0]-[W0]]_[Timestamp].json
    The directory can be named exactly the same.
    The summary can simply be written in Markdown. It will contain:
    - All starting conditions
    - Final counts
    - Final mean theta
    - Final max theta
    - Final min theta
    - Final median theta
    - Other statistics of interest
    - Ten intermediate states
    - Real time elapsed
    - Tokens and/or cost of the simulation

    The detailed results will contain:
     - A csv census file (called census.csv) with
        - populations for each step (w and s)
        - average theta for each step
    - A json file for each wolf with:
        - wolf id
        - birth step
        - death step
        - For every step:
            - theta
            - explanation
            - vocalization
            - prompt

    Args:
        results (dict): The simulation results dictionary, expected to include simulation history,
                        final counts, and optionally model and agents details.
        results_path (str or None): Path to save results. If None, a default filename with timestamp is used.
    """
    # Get current working directory (which will be the notebook directory if running from a notebook)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Use module directory as base
    # Use the provided path or default
    results_path = results_path if results_path else DEFAULT_RESULTS_PATH

    # If the path is not absolute, make it relative to the module directory
    if not os.path.isabs(results_path):
        path = os.path.abspath(os.path.join(current_dir, results_path))
    else:
        path = results_path

    # Create the results directory if it doesn't exist
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created results directory at: {path}")
    except OSError as e:
        print(f"Error creating directory {path}: {e}")
        # Fall back to a directory in the current working directory
        path = os.path.join(current_dir, "notebook_results")
        print(f"Falling back to: {path}")
        os.makedirs(path, exist_ok=True)

    # Construct a summary filename based on simulation parameters and current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_params = results.get("model_params", {})
    model_name = model_params.get("model_name", "Model")
    steps = model_params.get("steps", "steps")
    s0 = model_params.get("s_start", "S0")
    w0 = model_params.get("w_start", "W0")
    summary_name = f"{model_name}_{steps}_{s0}-{w0}_{timestamp}"
    summary_filename = os.path.join(path, f"{summary_name}.md")
    details_dir = os.path.join(path, summary_name, "_detailed_results")
    census_file = os.path.join(details_dir, "census.csv")
    wolves_dir = os.path.join(details_dir, "wolves")

    # Compute final theta statistics from the last snapshot in history
    last_snapshot = results.get("history", [])[-1] if results.get("history") else {}
    final_thetas = last_snapshot.get("thetas", [])
    if final_thetas:
        final_mean_theta = sum(final_thetas) / len(final_thetas)
        final_max_theta = max(final_thetas)
        final_min_theta = min(final_thetas)
        sorted_thetas = sorted(final_thetas)
        median_index = len(sorted_thetas) // 2
        final_median_theta = sorted_thetas[median_index]
    else:
        final_mean_theta = final_max_theta = final_min_theta = final_median_theta = None

    # Sample ten intermediate states evenly from history
    history = results.get("history", [])
    num_states = len(history)
    sample_states = []
    if num_states > 0:
        step = max(1, num_states // 10)
        sample_states = history[::step][:10]

    # Placeholder for real time elapsed and token/cost info
    real_time_elapsed = "not measured"
    tokens_cost = "not computed"

    summary_lines = [
        "# Simulation Summary",
        "",
        "**Starting Conditions:**",
        f"{results.get('model_params', {})}",
        "",
        "**Final Counts:**",
        f"Sheep: {results.get('final_sheep', 'N/A')}",
        f"Wolves: {results.get('final_wolves', 'N/A')}",
        "",
        "**Theta Statistics:**",
        f"Mean Theta: {final_mean_theta}",
        f"Max Theta: {final_max_theta}",
        f"Min Theta: {final_min_theta}",
        f"Median Theta: {final_median_theta}",
        "",
        "**Intermediate States (sampled):**",
    ]

    for state in sample_states:
        summary_lines.append(
            f"Step {state.get('step')}: Sheep={format4(state.get('sheep'))}, Wolves={state.get('wolves')}, Mean Theta={format4(state.get('mean_theta'))}"
        )

    summary_lines += [
        "",
        f"**Real Time Elapsed:** {real_time_elapsed}",
        f"**Token/Cost Info:** {tokens_cost}",
    ]

    # Write summary file
    with open(summary_filename, "w") as f:
        f.write("\n".join(summary_lines))

    # Create detailed results directory inside the results path
    os.makedirs(details_dir, exist_ok=True)

    # Save census file: census.csv
    import csv
    with open(census_file, "w", newline='') as csvfile:
        fieldnames = ['step', 'sheep', 'wolves', 'mean_theta']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in results.get('history', []):
            writer.writerow({
                'step': snapshot.get('step'),
                'sheep': round4(snapshot.get('sheep')),
                'wolves': round4(snapshot.get('wolves')),
                'mean_theta': round4(snapshot.get('mean_theta'))
            })

    # Save individual wolf details
    agents = results.get('agents', [])
    os.makedirs(wolves_dir, exist_ok=True)
    for wolf in agents:
        wolf_filename = os.path.join(wolves_dir, f"wolf_{wolf.get('wolf_id')}.json")
        wolf_data = {
            'wolf_id': wolf.get('wolf_id'),
            'born_at_step': wolf.get('born_at_step'),
            'died_at_step': wolf.get('died_at_step'),
            'thetas': wolf.get('thetas'),
            'explanations': wolf.get('explanations'),
            'vocalizations': wolf.get('vocalizations'),
            'prompts': wolf.get('prompts', [])
        }
        with open(wolf_filename, 'w') as wf:
            json.dump(wolf_data, wf, indent=4)

    print(f"Simulation summary saved to {summary_filename} and details in directory {details_dir}")
