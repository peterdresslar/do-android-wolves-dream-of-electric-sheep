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
def dx_dt(x, t, alpha, beta, gamma, delta):  # noqa: ARG001 I want the t there for the moment
    s, w = x
    ds_dt = alpha * s - beta * s * w
    dw_dt = -gamma * w + delta * beta * s * w
    return [ds_dt, dw_dt]


def get_reference_ODE(model_params, model_time):
    alpha = model_params["alpha"]
    beta = model_params["beta"]
    gamma = model_params["gamma"]
    delta = model_params["delta"]

    t_end = model_time["time"]
    times = np.linspace(0, t_end, model_time["tmax"])
    x0 = [model_params["s_start"], model_params["w_start"]]

    integration = odeint(
        dx_dt, x0, times, args=(alpha, beta, gamma, delta)
    )  # via cursor, verify this
    ode_df = pd.DataFrame(
        {"t": times, "s": round4(integration[:, 0]), "w": round4(integration[:, 1])}
    )
    return ode_df


#################################################################
# Simulation Telemetry
#################################################################
def save_simulation_results(results, results_path=None):
    """Save simulation parameters (starting conditions) and results information to a summary file and a directory with detailed results.

    Args:
        results (dict): The simulation results dictionary, expected to include simulation history,
                        final counts, and optionally model and agents details.
        results_path (str or None): Path to save results. If None, a default filename with timestamp is used.
    """
    # Get current working directory (which will be the notebook directory if running from a notebook)
    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )  # Use module directory as base
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
    starting_sheep = model_params.get("s_start", "S0")
    starting_wolves = model_params.get("w_start", "W0")

    # Create a unique run directory name
    run_dir_name = (
        f"{model_name}_{steps}_{starting_sheep}-{starting_wolves}_{timestamp}"
    )
    run_dir = os.path.join(path, run_dir_name)

    # Create the run directory
    os.makedirs(run_dir, exist_ok=True)

    # Place the summary file in the run directory instead of the root
    summary_filename = os.path.join(run_dir, "summary.md")

    # Create detailed results directory inside the run directory
    details_dir = os.path.join(run_dir, "_detailed_results")
    census_file = os.path.join(details_dir, "census.csv")
    wolves_dir = os.path.join(details_dir, "wolves")

    # Generate history if it doesn't exist
    if "history" not in results:
        # Create a synthetic history from available data
        history = []
        sheep_history = results.get("sheep_history", [])
        wolf_history = results.get("wolf_history", [])
        theta_history = results.get("average_theta_history", [])

        # Use the longest history as the basis
        max_steps = max(len(sheep_history), len(wolf_history), len(theta_history))

        for step in range(max_steps):
            snapshot = {
                "step": step,
                "sheep": sheep_history[step] if step < len(sheep_history) else None,
                "wolves": wolf_history[step] if step < len(wolf_history) else None,
                "mean_theta": (
                    theta_history[step] if step < len(theta_history) else None
                ),
                # We don't have individual thetas for each step, so this will be empty
                "thetas": [],
            }
            history.append(snapshot)

        results["history"] = history

    # Compute final theta statistics from the last snapshot in history or from agents data
    history = results.get("history", [])
    last_snapshot = history[-1] if history else {}

    # Try to get thetas from the last snapshot, or use current_thetas if available
    final_thetas = last_snapshot.get("thetas", [])

    # If we don't have thetas in the history, try to extract from agents
    if not final_thetas and "agents" in results:
        # Extract thetas from living wolves
        final_thetas = []
        for wolf in results["agents"]:
            if wolf.get("alive", False) and wolf.get("thetas"):
                final_thetas.append(wolf["thetas"][-1])

    if final_thetas:
        final_mean_theta = sum(final_thetas) / len(final_thetas)
        final_max_theta = max(final_thetas)
        final_min_theta = min(final_thetas)
        sorted_thetas = sorted(final_thetas)
        median_index = len(sorted_thetas) // 2
        final_median_theta = sorted_thetas[median_index]
    else:
        # If we still don't have thetas, use the average_theta_history if available
        theta_history = results.get("average_theta_history", [])
        if theta_history:
            final_mean_theta = theta_history[-1]
            final_max_theta = final_min_theta = final_median_theta = final_mean_theta
        else:
            final_mean_theta = final_max_theta = final_min_theta = (
                final_median_theta
            ) = None

    # Sample ten intermediate states evenly from history
    # If fewer than 10 states, just sample and report all
    num_states = len(history)
    sample_states = []
    if num_states > 0:
        if num_states < 10:
            sample_states = history
        else:
            step = max(1, num_states // 10)
            sample_states = history[::step][:10]

    # Use runtime from results if available
    real_time_elapsed = f"{results.get('runtime', 'not measured')} seconds"
    tokens_cost = "not computed"

    usage = results.get("usage", {})  # sent with to_dict()

    summary_lines = [
        "# Simulation Summary",
        "",
        "**Starting Conditions:**",
        f"{results.get('model_params', {})}",
        "",
        "**Runtime and Usage:**",
        f"Runtime: {real_time_elapsed}",
        f"Usage: {usage}",
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
        sheep_val = state.get("sheep")
        wolves_val = state.get("wolves")
        theta_val = state.get("mean_theta")

        # Format values if they exist
        sheep_str = format4(sheep_val) if sheep_val is not None else "N/A"
        wolves_str = str(wolves_val) if wolves_val is not None else "N/A"
        theta_str = format4(theta_val) if theta_val is not None else "N/A"

        summary_lines.append(
            f"Step {state.get('step')}: Sheep={sheep_str}, Wolves={wolves_str}, Mean Theta={theta_str}"
        )

    summary_lines += [
        "",
        f"**Real Time Elapsed:** {real_time_elapsed}",
        f"**Token/Cost Info:** {tokens_cost}",
    ]

    # Write summary file
    with open(summary_filename, "w") as f:
        f.write("\n".join(summary_lines))

    # Create detailed results directory inside the run directory
    os.makedirs(details_dir, exist_ok=True)

    # Save census file: census.csv
    import csv

    with open(census_file, "w", newline="") as csvfile:
        fieldnames = ["step", "sheep", "wolves", "mean_theta"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in history:
            writer.writerow(
                {
                    "step": snapshot.get("step"),
                    "sheep": (
                        round4(snapshot.get("sheep"))
                        if snapshot.get("sheep") is not None
                        else None
                    ),
                    "wolves": (
                        round4(snapshot.get("wolves"))
                        if snapshot.get("wolves") is not None
                        else None
                    ),
                    "mean_theta": (
                        round4(snapshot.get("mean_theta"))
                        if snapshot.get("mean_theta") is not None
                        else None
                    ),
                }
            )

    # Save individual wolf details
    agents = results.get("agents", [])
    os.makedirs(wolves_dir, exist_ok=True)
    for wolf in agents:
        # Use wolf_id if id is not available
        wolf_id = wolf.get("wolf_id", wolf.get("id", "unknown"))
        wolf_filename = os.path.join(wolves_dir, f"wolf_{wolf_id}.json")

        # Create wolf data structure - use the flattened structure directly
        wolf_data = {
            "wolf_id": wolf_id,
            "beta": wolf.get("beta"),
            "gamma": wolf.get("gamma"),
            "delta": wolf.get("delta"),
            "alive": wolf.get("alive"),
            "born_at_step": wolf.get("born_at_step"),
            "died_at_step": wolf.get("died_at_step"),
            "thetas": wolf.get("thetas", []),
            "decision_history": {
                "history_steps": wolf.get("history_steps", []),
                "new_thetas": wolf.get("new_thetas", []),
                "prompts": wolf.get("prompts", []),
                "explanations": wolf.get("explanations", []),
                "vocalizations": wolf.get("vocalizations", []),
            },
        }

        with open(wolf_filename, "w") as wf:
            json.dump(wolf_data, wf, indent=4)

    print(f"Simulation results saved to {run_dir}")
    print(f"Summary: {summary_filename}")
    print(f"Detailed results: {details_dir}")
