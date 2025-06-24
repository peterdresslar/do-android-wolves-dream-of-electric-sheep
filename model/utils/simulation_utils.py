# simulation_utils.py
"""
Mostly reference ODE functions
"""

import datetime
import json
import os
import re

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from scipy.integrate import solve_ivp


# Function to get the project root directory
def get_project_root():
    """Return the path to the project root directory."""
    # Start from the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels (from model/utils to project root)
    return os.path.abspath(os.path.join(current_dir, "../.."))


# Define results path relative to project root
DEFAULT_RESULTS_PATH = os.path.join(get_project_root(), "data", "results")


# Function to resolve paths relative to project root
def resolve_path(path):
    """
    With any relative path, resolve it to the results directory,
    which, relative to the project root, is "data/results".
    """
    if os.path.isabs(path):
        return path
    return os.path.join(DEFAULT_RESULTS_PATH, path)


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
def dx_dt(x, t, alpha, beta, gamma, delta):  # noqa
    s, w = x
    ds_dt = alpha * s - beta * s * w
    dw_dt = -gamma * w + delta * s * w
    return [ds_dt, dw_dt]


def get_reference_ODE_with_cliff(model_params, model_time, cliff_type: str = "none"):
    """
    Get the reference ODE solution using solve_ivp with event detection for extinction.
    The cliff type can be "none", "sheep", "wolves", or "both".
    """
    alpha = model_params["alpha"]
    beta = model_params["beta"]
    gamma = model_params["gamma"]
    delta = model_params["delta"]
    eps = model_params.get("eps", 0.0)

    t_end = model_time["time"]
    t_eval = np.linspace(0, t_end, model_time["tmax"] + 1)

    # Define the ODE system
    def dx_dt_ivp(t, x):
        s, w = x
        # If populations are extinct (below 1), keep them at 0
        if cliff_type in ["sheep", "both"] and s < 1:
            s = 0
        if cliff_type in ["wolves", "both"] and w < 1:
            w = 0

        # the LV susyem
        ds_dt = alpha * s - beta * s * w
        dw_dt = -gamma * w + delta * s * w
        return [ds_dt, dw_dt]

    # Define extinction events
    events = []

    if cliff_type in ["sheep", "both"]:

        def sheep_extinction(t, x):
            return x[0] - 1  # Triggers when sheep population crosses 1

        sheep_extinction.terminal = False
        sheep_extinction.direction = -1  # Only trigger when crossing from above
        events.append(sheep_extinction)

    if cliff_type in ["wolves", "both"]:

        def wolf_extinction(t, x):
            return x[1] - 1  # Triggers when wolf population crosses 1

        wolf_extinction.terminal = False
        wolf_extinction.direction = -1
        events.append(wolf_extinction)

    # Initial conditions
    x0 = [model_params["s_start"], model_params["w_start"]]

    # Solve the ODE
    solution = solve_ivp(
        dx_dt_ivp,
        [0, t_end],
        x0,
        t_eval=t_eval,
        events=events,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )

    # Post-process to ensure extinct populations stay at 0
    s_values = solution.y[0]
    w_values = solution.y[1]

    # Find when extinctions occurred
    if cliff_type in ["sheep", "both"]:
        sheep_extinct_idx = np.where(s_values < 1)[0]
        if len(sheep_extinct_idx) > 0:
            s_values[sheep_extinct_idx[0] :] = 0

    if cliff_type in ["wolves", "both"]:
        wolf_extinct_idx = np.where(w_values < 1)[0]
        if len(wolf_extinct_idx) > 0:
            w_values[wolf_extinct_idx[0] :] = 0

    ode_df = pd.DataFrame(
        {
            "t": solution.t,
            "s": np.round(s_values, 4),
            "w": np.round(w_values, 4),
        }
    )

    return ode_df


#################################################################
# Plotting
#################################################################
def create_population_plot(results, sheep_max, title=None) -> plt.Figure:
    """Create a plot of the population over time."""
    # Create a proper time steps array
    sheep_history = results.get("sheep_history", [])
    wolf_history = results.get("wolf_history", [])
    theta_history = results.get("average_theta_history", [])

    # Ensure all histories have the same length by padding with the last value
    max_length = max(len(sheep_history), len(wolf_history), len(theta_history))

    # Pad arrays if needed
    if len(sheep_history) < max_length:
        last_value = sheep_history[-1] if sheep_history else 0
        sheep_history = sheep_history + [last_value] * (max_length - len(sheep_history))

    if len(wolf_history) < max_length:
        last_value = wolf_history[-1] if wolf_history else 0
        wolf_history = wolf_history + [last_value] * (max_length - len(wolf_history))

    if len(theta_history) < max_length:
        last_value = theta_history[-1] if theta_history else 0
        theta_history = theta_history + [last_value] * (max_length - len(theta_history))

    time_steps = list(range(max_length))

    ai_df = pd.DataFrame(
        {
            "t": time_steps,
            "Sheep": sheep_history,
            "Wolves": wolf_history,
            "Avg Theta": theta_history,
        }
    )

    # Create a plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot populations on left y-axis
    sns.lineplot(
        data=pd.melt(ai_df, id_vars=["t"], value_vars=["Sheep", "Wolves"]),
        x="t",
        y="value",
        hue="variable",
        palette=["cadetblue", "darkred"],
        ax=ax1,
    )

    # Set the left y-axis limit to include a small margin
    ax1.set_ylim(0, sheep_max * 1.1)

    # Plot average theta on right y-axis
    ax2 = ax1.twinx()
    sns.lineplot(
        data=ai_df,
        x="t",
        y="Avg Theta",
        color="darkgreen",
        ax=ax2,
        linewidth=2,
    )

    # Set the y-axis limits for the right axis to align with the left axis

    ax2.set_ylim(0, 1.1)

    # Add labels and legend
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Population")
    ax2.set_ylabel("Average Theta")
    ax1.legend(title="", loc="upper left", frameon=False)
    ax2.legend(["Avg Theta"], loc="upper right", frameon=False)

    plt.title(title)
    plt.tight_layout()

    return fig


def create_replot(path, width=12, dpi=100):
    """
    Replot the population plot from a given path to a census.csv file.

    Args:
        path (str): Path to the census.csv file or directory containing it
        width (int): Width of the plot in inches (default: 12)
        dpi (int): Dots per inch for the figure (default: 100)

    Returns:
        plt.Figure: The generated figure
    """
    # Handle if path is a directory by looking for census.csv inside
    if os.path.isdir(path):
        # Check for detailed_results subdirectory
        detailed_path = os.path.join(path, "_detailed_results")
        if os.path.exists(detailed_path):
            census_path = os.path.join(detailed_path, "census.csv")
        else:
            # Look for census.csv directly in the provided directory
            census_path = os.path.join(path, "census.csv")
    else:
        # Assume path is directly to the census file
        census_path = path

    # Check if the file exists
    if not os.path.exists(census_path):
        raise FileNotFoundError(f"Census file not found at {census_path}")

    # Read the census data
    df = pd.read_csv(census_path)

    # If width is very large (>100), assume it's in pixels and convert to inches
    if width > 100:
        width_inches = width / dpi
        height_inches = (width * 7 / 12) / dpi
    else:
        width_inches = width
        height_inches = width * 7 / 12

    # Create a plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(width_inches, height_inches), dpi=dpi)

    # Plot populations on left y-axis
    sns.lineplot(
        data=pd.melt(df, id_vars=["step"], value_vars=["sheep", "wolves"]),
        x="step",
        y="value",
        hue="variable",
        palette=["cadetblue", "darkred"],
        ax=ax1,
    )

    # Find the maximum population value to align axes properly
    max_pop_value = max(df["sheep"].max(), df["wolves"].max())

    # Set the left y-axis limit to include a small margin
    ax1.set_ylim(0, max_pop_value * 1.1)

    # Plot average theta on right y-axis
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x="step",
        y="mean_theta",
        color="darkgreen",
        ax=ax2,
        linewidth=2,
    )

    # Set the y-axis limits for the right axis to always be 0-1 for theta
    # but scale it to align with the population axis
    ax2.set_ylim(0, 1)

    # This makes theta y-axis visually match the population y-axis
    ax2.set_ylim(0, 1 * (max_pop_value * 1.1))

    # Add labels and legend
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Population")
    ax2.set_ylabel("Average Theta")
    ax1.legend(["Sheep", "Wolves"], loc="upper left", frameon=False)
    ax2.legend(["Avg Theta"], loc="upper right", frameon=False)

    # Try to extract information for the title from the path
    try:
        # Get the directory name which contains the run information
        if os.path.isdir(path):
            dir_name = os.path.basename(path)
        else:
            dir_name = os.path.basename(os.path.dirname(path))

        # Parse the directory name to extract model and prompt type
        parts = dir_name.split("_")
        if len(parts) >= 4:
            model_name = parts[0]
            prompt_type = parts[3] if parts[3] not in ["20", "19", "18"] else "None"

            if prompt_type == "None":
                # Check if there's a decision_mode and theta_start value in the summary file
                summary_path = os.path.join(
                    os.path.dirname(census_path), "..", "summary.md"
                )
                if os.path.exists(summary_path):
                    with open(summary_path) as f:
                        summary_text = f.read()
                        # Look for decision_mode
                        decision_mode_match = re.search(
                            r"decision_mode['\"]?: ?['\"]?([a-z]+)['\"]?", summary_text
                        )
                        decision_mode = (
                            decision_mode_match.group(1)
                            if decision_mode_match
                            else "ai"
                        )

                        if decision_mode == "constant":
                            # Look for theta_start
                            theta_match = re.search(
                                r"theta_start['\"]?: ?([0-9.]+)", summary_text
                            )
                            if theta_match:
                                theta_start = float(theta_match.group(1))
                                title = f"Population Dynamics with Constant Theta Value. Theta = {theta_start}"
                            else:
                                title = "Population Dynamics with Constant Theta"
                        elif decision_mode == "adaptive":
                            # Look for k value
                            k_match = re.search(r"k['\"]?: ?([0-9.]+)", summary_text)
                            k_value = float(k_match.group(1)) if k_match else 1.0
                            title = f"Population Dynamics with Adaptive Theta Function. Sensitivity = {k_value}"
                        else:
                            title = (
                                "Population Dynamics with AI-determined theta values"
                            )
                else:
                    title = "Population Dynamics with Algorithmic Theta Function"
            else:
                title = f"Population Dynamics with AI-determined theta values. Model: {model_name}, Prompt Type: {prompt_type} information."
        else:
            title = "Population Dynamics"
    finally:
        plt.title(title)
        plt.tight_layout()

    return fig


def save_replot(path, output_path=None, width=12):
    """
    Create and save a replot from census data

    Args:
        path (str): Path to the census.csv file or directory containing it
        output_path (str): Path to save the plot (default: same directory as census with name 'replot.png')
        width (int): Width of the plot in inches

    Returns:
        str: Path to the saved plot
    """
    fig = create_replot(path, width)

    # Determine output path
    if output_path is None:
        if os.path.isdir(path):
            output_path = os.path.join(path, "replot.png")
        else:
            output_path = os.path.join(os.path.dirname(path), "replot.png")

    # Save the figure
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return output_path


def format_output(text):
    """
    Format text to be saved in a JSON file. Get rid of any problem characters.
    """
    if text is None:
        return ""
    if isinstance(text, list):
        return [format_output(item) for item in text]
    elif isinstance(text, dict):
        return {key: format_output(value) for key, value in text.items()}
    else:
        return text.replace("\n", "<br>")


#################################################################
# Simulation Telemetry
#################################################################
def save_simulation_results(results, results_path=None):
    """Save simulation parameters (starting conditions) and results information to a summary file and a directory with detailed results.

    For example, if we do not get a path, we set up a results base directory relative to DEFAULT_RESULTS_PATH
    based on the model_name, steps, starting_sheep, starting_wolves, prompt_type, and timestamp.

    If we do get a path, that path, relative to the DEFAULT_RESULTS_PATH, is used as the base directory.

    Args:
        results (dict): The simulation results dictionary, expected to include simulation history,
                        final counts, and optionally model and agents details.
        results_path (str or None): Path to save results. If None, a default filename with timestamp is used.
    """
    # If no path is provided, use DEFAULT_RESULTS_PATH directly
    if results_path is None:
        path = DEFAULT_RESULTS_PATH
    else:
        # If a path is provided, resolve it relative to DEFAULT_RESULTS_PATH
        # unless it's an absolute path
        if os.path.isabs(results_path):
            path = results_path
        else:
            path = os.path.join(DEFAULT_RESULTS_PATH, results_path)

    # Create the results directory if it doesn't exist
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}")
        # Fall back to a directory in the current working directory
        fallback_path = os.path.join(get_project_root(), "notebook_results")
        print(f"Falling back to: {fallback_path}")
        os.makedirs(fallback_path, exist_ok=True)
        path = fallback_path

    # Construct a summary filename based on simulation parameters and current timestamp
    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M"
    )  # Shortened timestamp (removed seconds)
    model_params = results.get("model_params", {})
    model_opts = results.get("model_opts", {})

    print("model_opts", model_opts)
    print("model_params", model_params)

    model_name = model_params.get("model_name")

    prompt_type = model_params.get("prompt_type")
    temperature = model_params.get("temperature")  # Default to 0.2 if not specified
    steps = model_params.get("steps", "steps")
    starting_sheep = model_params.get("s_start")
    starting_wolves = model_params.get("w_start")

    print("model_name", model_name)
    print("prompt_type", prompt_type)
    print("temperature", temperature)

    # Create a unique run directory name
    # Include temperature in directory name only for AI mode
    decision_mode = model_params.get("decision_mode")

    if decision_mode in ("adaptive", "constant"):
        # For non-AI modes, simplify naming

        print("steps", steps)
        print("starting_sheep", starting_sheep)
        print("starting_wolves", starting_wolves)
        print("decision_mode", decision_mode)
        print("timestamp", timestamp)
        run_dir_name = (
            f"{steps}_{starting_sheep}-{starting_wolves}_{decision_mode}_{timestamp}"
        )
    else:  # AI mode
        # Format temperature with only one decimal place and keep model name shorter
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        model_short = model_short[:8]  # Limit to first 8 chars of model name
        temp_str = f"{temperature:.1f}".replace(".0", "")
        # Use shorter prompt identifier
        prompt_short = prompt_type[:4] if prompt_type else "none"
        run_dir_name = f"{model_short}_{steps}_{starting_sheep}-{starting_wolves}_{prompt_short}_t{temp_str}_{timestamp}"

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
    # format usage.cost to 4 decimal places
    if usage.get("cost"):
        usage["cost"] = round4(usage.get("cost"))
    else:
        usage["cost"] = "not computed"

    summary_lines = [
        "# Simulation Summary",
        "",
        "**Starting Conditions:**",
        f"{results.get('model_params', {})}",
        "",
        "**Runtime and Usage:**",
        f"Runtime: {real_time_elapsed}",
        f"Usage: {usage}",
        f"Prompt Type: {prompt_type}",
        f"Temperature: {temperature}",
        "",
        "**Final Counts:**",
        f"Sheep: {results.get('final_sheep')}",
        f"Wolves: {results.get('final_wolves')}",
        "",
        "**Step of Last Wolf Death:**",
        f"{results.get('last_wolf_death_step')}",
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
        sheep_str = format4(sheep_val)
        wolves_str = str(wolves_val)
        theta_str = format4(theta_val)

        summary_lines.append(
            f"Step {state.get('step')}: Sheep={sheep_str}, Wolves={wolves_str}, Mean Theta={theta_str}"
        )

    # Add to results for later retrieval
    results["final_wolves"] = (
        results["wolf_history"][-1] if results.get("wolf_history", []) else 0
    )

    # Add this information to the summary file
    if results.get("last_wolf_death_step", None) is not None:
        summary_lines.append(
            f"**Step of Last Wolf Death:** {results.get('last_wolf_death_step')}"
        )
    else:
        if results.get("wolf_history", []) and results["wolf_history"][-1] > 0:
            summary_lines.append("**Step of Last Wolf Death:** N/A (wolves survived)")
        else:
            summary_lines.append("**Step of Last Wolf Death:** N/A (no data)")

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
        # Ensure all lists in decision_history have the same length by filling with None/null values
        history_steps = wolf.get("history_steps", [])
        new_thetas = wolf.get("new_thetas", [])
        prompts = wolf.get("prompts", [])
        explanations = wolf.get("explanations", [])
        vocalizations = wolf.get("vocalizations", [])

        # Find the maximum length of any of these lists
        max_length = max(
            len(history_steps),
            len(new_thetas),
            len(prompts),
            len(explanations),
            len(vocalizations),
        )

        # Ensure all lists have the same length by padding with None
        history_steps = history_steps + [None] * (max_length - len(history_steps))
        new_thetas = new_thetas + [None] * (max_length - len(new_thetas))
        prompts = prompts + [None] * (max_length - len(prompts))
        explanations = explanations + [None] * (max_length - len(explanations))
        vocalizations = vocalizations + [None] * (max_length - len(vocalizations))

        wolf_data = {
            "wolf_id": wolf_id,
            "beta": wolf.get("beta"),
            "gamma": wolf.get("gamma"),
            "delta": wolf.get("delta"),
            "alive": wolf.get("alive"),
            "born_at_step": wolf.get("born_at_step"),
            "died_at_step": wolf.get("died_at_step"),
            "thetas": wolf.get("thetas", []),
            "decision_history": {  # format prompts, explanations, and vocalizations in case they have problem characters
                "history_steps": history_steps,
                "new_thetas": new_thetas,
                "prompts": [format_output(prompt) for prompt in prompts],
                "explanations": [
                    format_output(explanation) for explanation in explanations
                ],
                "vocalizations": [
                    format_output(vocalization) for vocalization in vocalizations
                ],
            },
        }

        with open(wolf_filename, "w") as wf:
            json.dump(wolf_data, wf, indent=4)

    # finally, write a plot and save it to the run_dir
    # donʻt display it.

    decision_mode = model_params.get("decision_mode")
    if decision_mode == "adaptive":
        title = f"Population Dynamics with Adaptive Theta Function. Sensitivity = {model_params.get('k')}"
    elif decision_mode == "constant":
        title = f"Population Dynamics with Constant Theta Value. Theta = {model_params.get('theta_start')}"
    else:  # AI mode
        title = f"Population Dynamics with AI-determined theta values. Model: {model_name}, Prompt Type: {prompt_type}, Temperature: {temperature}."

    fig = create_population_plot(results, model_params.get("sheep_max"), title=title)
    fig.savefig(os.path.join(run_dir, "population_plot.png"))

    if model_opts.get("step_print", False):
        print(f"Simulation results saved to {run_dir}")
