#!/usr/bin/env python3
# run_multi_experiment.py
import argparse
import concurrent.futures
import datetime
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import the run function directly from model.py
from model.model import run
from model.utils.simulation_utils import get_project_root, round4


def load_presets(presets_file="presets.json"):
    """
    Load simulation presets from a JSON file.

    Args:
        presets_file (str): Path to the presets JSON file

    Returns:
        dict: Dictionary with preset names as keys and preset configurations as values
    """
    try:
        # Try to find the presets file relative to the project root
        project_root = get_project_root()
        presets_path = os.path.join(project_root, presets_file)

        with open(presets_path) as f:
            data = json.load(f)

        # Convert the list of presets to a dictionary for easier access
        presets_dict = {}
        for preset in data.get("presets", []):
            preset_name = preset.get("preset_name")
            if preset_name:
                presets_dict[preset_name] = preset

        return presets_dict
    except Exception as e:
        print(f"Error loading presets file: {e}")
        return {}


def generate_sweep_configs(preset):
    """
    Generate a list of configurations for a parameter sweep.

    Supports one or two sweep variables as specified in the preset's 'sweep_variables' list.
    For each variable, there should be a corresponding list of values in 'sweep_parameters'
    with the key format '{variable}_values'. Variables are run in order inside-outside.

    Args:
        preset (dict): Preset configuration with sweep_variables, sweep_parameters, and fixed_parameters

    Returns:
        list: List of configuration dictionaries for each combination of sweep parameters
    """
    fixed_params = preset.get("fixed_parameters", {}).copy()
    sweep_params = preset.get("sweep_parameters", {})
    sweep_variables = preset.get("sweep_variables", [])

    # Validate sweep variables
    if not sweep_variables:
        print("Error: No sweep variables specified in preset")
        return []

    if len(sweep_variables) > 2:
        print(
            "Warning: More than two sweep variables specified. Only the first two will be used."
        )
        sweep_variables = sweep_variables[:2]

    # Validate that we have values for each sweep variable
    for var in sweep_variables:
        values_key = f"{var}_values"
        if values_key not in sweep_params:
            print(
                f"Error: No values found for sweep variable '{var}' (expected key '{values_key}')"
            )
            return []

    configs = []

    # Get the base path without "data/results" prefix to avoid doubling
    base_path = fixed_params.get("path", "")
    # Remove "data/results/" prefix if it exists to avoid path doubling
    if base_path.startswith("data/results/"):
        base_path = base_path[len("data/results/") :]

    # Handle single variable sweep
    if len(sweep_variables) == 1:
        var = sweep_variables[0]
        values_key = f"{var}_values"
        values = sweep_params[values_key]

        for value in values:
            # Start with fixed parameters
            config = fixed_params.copy()

            # Add sweep parameter
            config[var] = value

            # Create a unique path for this configuration
            config["path"] = (
                f"{base_path}/{var}_{value}" if base_path else f"{var}_{value}"
            )

            configs.append(config)

    # Handle two variable sweep
    elif len(sweep_variables) == 2:
        var1 = sweep_variables[0]
        var2 = sweep_variables[1]
        values1_key = f"{var1}_values"
        values2_key = f"{var2}_values"
        values1 = sweep_params[values1_key]
        values2 = sweep_params[values2_key]

        for value1 in values1:
            for value2 in values2:
                # Start with fixed parameters
                config = fixed_params.copy()

                # Add sweep parameters
                config[var1] = value1
                config[var2] = value2

                # Create a unique path for this configuration
                config["path"] = (
                    f"{base_path}/{var1}_{value1}_{var2}_{value2}"
                    if base_path
                    else f"{var1}_{value1}_{var2}_{value2}"
                )

                configs.append(config)

    print(f"Generated {len(configs)} configurations for sweep")
    return configs


def generate_prompt_sweep_configs(preset):
    """
    Generate a list of configurations for a prompt sweep.

    For each value in the sweep variable, creates configurations for:
    1. Each prompt_type (high, medium, low) with decision_mode="ai"
    2. An adaptive theta run (decision_mode="adaptive") with k parameter

    The constant theta mode is no longer included.

    Args:
        preset (dict): Preset configuration with sweep_variables, sweep_parameters, and fixed_parameters

    Returns:
        list: List of configuration dictionaries for each combination
    """
    fixed_params = preset.get("fixed_parameters", {}).copy()
    sweep_params = preset.get("sweep_parameters", {})
    sweep_variables = preset.get("sweep_variables", [])

    # Validate sweep variables - prompt sweeps should have exactly one sweep variable
    if not sweep_variables:
        print("Error: No sweep variables specified in preset")
        return []

    if len(sweep_variables) > 1:
        print(
            "Warning: Prompt sweeps work best with one sweep variable. Using only the first one."
        )
        sweep_variables = sweep_variables[:1]

    var = sweep_variables[0]
    values_key = f"{var}_values"

    # Validate that we have values for the sweep variable
    if values_key not in sweep_params:
        print(
            f"Error: No values found for sweep variable '{var}' (expected key '{values_key}')"
        )
        return []

    values = sweep_params[values_key]

    # Get the base path without "data/results" prefix to avoid doubling
    base_path = fixed_params.get("path", "")
    if base_path.startswith("data/results/"):
        base_path = base_path[len("data/results/") :]

    # Define prompt types to use
    prompt_types = ["high", "medium", "low"]

    configs = []

    # For each value in the sweep variable
    for value in values:
        # For each prompt type
        for prompt_type in prompt_types:
            # Start with fixed parameters
            config = fixed_params.copy()

            # Add sweep parameter and prompt type
            config[var] = value
            config["prompt_type"] = prompt_type
            config["decision_mode"] = "ai"  # Ensure AI is enabled for prompt runs

            # Create a unique path for this configuration
            config["path"] = f"{base_path}/{var}_{value}_prompt_{prompt_type}"

            configs.append(config)

        # Add an adaptive theta run with the same parameters
        config = fixed_params.copy()
        config[var] = value
        config["decision_mode"] = "adaptive"
        k_value = config.get("k")
        config["path"] = f"{base_path}/{var}_{value}_theta_k_{k_value}"

        configs.append(config)

    print(f"Generated {len(configs)} configurations for prompt sweep")
    return configs


def run_simulation(config):
    """Run a single simulation with the given configuration."""
    print(f"Running simulation with config: {config}")

    try:
        # Call the run function directly
        start_time = time.time()
        results = run(**config)
        end_time = time.time()

        print(
            f"Simulation completed in {round4(end_time - start_time)} seconds: {config.get('path', 'unknown')}"
        )

        # Add runtime to results
        results["runtime"] = end_time - start_time

        # Check if we have wolf history data
        if not results.get("wolf_history"):
            print(
                f"ERROR: No wolf history data generated for {config.get('path', 'unknown')}"
            )
            return False, config, {"error": "No wolf history data generated"}

        # REMOVE OR MODIFY the initial wolf count check, as wolves might die in the first step
        # for certain parameter combinations
        # Instead of failing the simulation, just log a warning
        initial_wolves = results.get("wolf_history", [0])[0]
        expected_wolves = config.get("w_start")
        if initial_wolves != expected_wolves:
            print(
                f"WARNING: Initial wolf count mismatch in {config.get('path', 'unknown')}. Expected {expected_wolves}, got {initial_wolves}. This is normal for some parameter combinations where wolves die immediately."
            )
            # Continue with the simulation instead of failing

        return True, config, results
    except Exception as e:
        import traceback

        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "config": {k: v for k, v in config.items() if k != "path"},
        }
        print(f"Error running simulation: {str(e)}\n{traceback.format_exc()}")
        return False, config, error_details


def create_sweep_visualization(sweep_stats, results, preset, output_dir):
    """
    Create a grid visualization of simulation results for parameter sweeps.

    For each parameter combination, creates a small plot showing wolf population,
    sheep population, and average theta over time. Plots are arranged in a grid
    with rows and columns labeled by parameter values.

    Args:
        sweep_stats (list): List of statistics for each sweep configuration
        results (list): List of detailed results for each simulation
        preset (dict): The preset configuration used for the sweep
        output_dir (str): Directory to save the visualization

    Returns:
        str: Path to the saved visualization file
    """
    sweep_variables = preset.get("sweep_variables", [])
    if not sweep_variables or not sweep_stats:
        print("No sweep variables or results to visualize")
        return None

    # Extract unique values for each sweep variable
    unique_values = {}
    for var in sweep_variables:
        unique_values[var] = sorted({stat["config"].get(var) for stat in sweep_stats})

    # Determine grid dimensions
    if len(sweep_variables) == 1:
        # Single variable sweep - horizontal layout
        var = sweep_variables[0]
        n_cols = len(unique_values[var])
        n_rows = 1
        grid_positions = {
            (0, i): {var: val} for i, val in enumerate(unique_values[var])
        }
    elif len(sweep_variables) == 2:
        # Two variable sweep - grid layout
        var1, var2 = sweep_variables
        n_rows = len(unique_values[var1])
        n_cols = len(unique_values[var2])
        grid_positions = {
            (i, j): {var1: val1, var2: val2}
            for i, val1 in enumerate(unique_values[var1])
            for j, val2 in enumerate(unique_values[var2])
        }
    else:
        print("Cannot visualize more than two sweep variables")
        return None

    # Find the maximum sheep capacity across all simulations
    sheep_max = 0
    for result_entry in results:
        if not result_entry["success"]:
            continue
        sim_results = result_entry["results"]
        sheep_history = sim_results.get("sheep_history", [])
        if sheep_history:
            sheep_max = max(sheep_max, max(sheep_history))

    # Check if we should fit y-axis to sheep_max for each subplot
    fit_sheep_max_axis = preset.get("fixed_parameters", {}).get("fit-sheep-max-axis", False)

    # Create the figure with appropriate size
    # Base size of 3 inches per plot, with extra space for labels
    fig_width = max(8, 2 + 2 * n_cols)
    fig_height = max(6, 2 + 2 * n_rows)
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create GridSpec for the main grid and labels
    gs = GridSpec(n_rows + 1, n_cols + 1, figure=fig)

    # Create the plots
    for (row, col), config in grid_positions.items():
        # Find the matching result for this configuration
        matching_result = None
        for result_entry in results:
            if not result_entry["success"]:
                continue

            entry_config = result_entry["config"]
            matches = all(entry_config.get(var) == val for var, val in config.items())

            if matches:
                matching_result = result_entry
                break

        if not matching_result:
            continue

        sim_results = matching_result["results"]
        entry_config = matching_result["config"]  # Get the full config for this entry

        # Get history data
        sheep_history = sim_results.get("sheep_history", [])
        wolf_history = sim_results.get("wolf_history", [])
        theta_history = sim_results.get("average_theta_history", [])

        # Create subplot
        ax = fig.add_subplot(gs[row + 1, col + 1])

        # Plot data if available
        steps = range(len(sheep_history))
        if sheep_history:
            ax.plot(steps, sheep_history, color="cadetblue", linewidth=1)
        if wolf_history:
            ax.plot(steps, wolf_history, color="darkred", linewidth=1)

        # Create a twin axis for theta
        if theta_history:
            ax2 = ax.twinx()

            # Create a list of x/y points where wolves exist
            valid_x = []
            valid_theta = []
            for i, wolves in enumerate(wolf_history):
                if i < len(theta_history) and wolves > 0:
                    valid_x.append(i)
                    valid_theta.append(theta_history[i])

            # Only plot theta where wolves exist
            if valid_x and valid_theta:
                ax2.plot(valid_x, valid_theta, color="darkgreen", linewidth=1)
            ax2.set_ylim(0, 1)
            ax2.axis("off")  # Hide the second y-axis

        # Set y-limit for sheep and wolves
        if fit_sheep_max_axis and "sheep_max" in entry_config:
            # Use the sheep_max from the configuration for this specific subplot
            subplot_sheep_max = entry_config.get("sheep_max")
            ax.set_ylim(0, subplot_sheep_max * 1.1)  # Add 10% margin
        else:
            # Use the global maximum sheep population
            ax.set_ylim(0, sheep_max * 1.1)  # Add 10% margin

        # Remove ticks and labels for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # Add row labels (first sweep variable)
    if len(sweep_variables) >= 1 and n_rows > 1:
        var1 = sweep_variables[0]
        for i, val in enumerate(unique_values[var1]):
            ax = fig.add_subplot(gs[i + 1, 0])
            ax.text(0.5, 0.5, f"{var1}={val}", ha="center", va="center", rotation=90)
            ax.axis("off")

    # Add column labels (second sweep variable or first if only one)
    label_var = sweep_variables[1] if len(sweep_variables) > 1 else sweep_variables[0]
    for j, val in enumerate(unique_values[label_var]):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.text(0.5, 0.5, f"{label_var}={val}", ha="center", va="center")
        ax.axis("off")

    # Add a title + \n + subtitle
    plot_title = f"{preset.get('preset_name', 'Parameter Sweep')}\n{preset.get('preset_description', '')}"
    plt.suptitle(plot_title)

    # Add a small legend in the top-left corner
    legend_ax = fig.add_subplot(gs[0, 0])
    legend_ax.plot([], [], color="cadetblue", label="Sheep")
    legend_ax.plot([], [], color="darkred", label="Wolves")
    legend_ax.plot([], [], color="darkgreen", label="Theta")
    legend_ax.legend(loc="center", frameon=False, fontsize="small")
    legend_ax.axis("off")

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the title

    # Save the figure
    output_path = os.path.join(output_dir, "sweep_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def create_prompt_sweep_visualization(sweep_stats, results, preset, output_dir):
    """
    Create a grid visualization of simulation results for prompt sweeps.

    Arranges plots in a grid where:
    - Columns are the different prompt types (high, medium, low) and the adaptive theta function
    - Rows are the values of the sweep parameter

    Args:
        sweep_stats (list): List of statistics for each sweep configuration
        results (list): List of detailed results for each simulation
        preset (dict): The preset configuration used for the sweep
        output_dir (str): Directory to save the visualization

    Returns:
        str: Path to the saved visualization file
    """
    sweep_variables = preset.get("sweep_variables", [])
    if not sweep_variables or not sweep_stats:
        print("No sweep variables or results to visualize")
        return None

    # We expect exactly one sweep variable for prompt sweeps
    var = sweep_variables[0]

    # Extract unique values for the sweep variable
    unique_values = sorted({stat["config"].get(var) for stat in sweep_stats})

    # Define the column order: prompt types (high, medium, low) + adaptive theta
    column_types = ["high", "medium", "low", "adaptive"]

    # Determine grid dimensions
    n_rows = len(unique_values)
    n_cols = len(column_types)

    # Create a mapping of grid positions to configurations
    grid_positions = {}
    for i, val in enumerate(unique_values):
        for j, col_type in enumerate(column_types):
            grid_positions[(i, j)] = {"var": var, "value": val, "type": col_type}

    # Find the maximum sheep capacity across all simulations
    sheep_max = 0
    for result_entry in results:
        if not result_entry["success"]:
            continue
        sim_results = result_entry["results"]
        sheep_history = sim_results.get("sheep_history", [])
        if sheep_history:
            sheep_max = max(sheep_max, max(sheep_history))

    # Check if we should fit y-axis to sheep_max for each subplot
    fit_sheep_max_axis = preset.get("fixed_parameters", {}).get("fit-sheep-max-axis", False)

    # Create the figure with appropriate size
    fig_width = max(12, 2 + 2.5 * n_cols)
    fig_height = max(8, 2 + 1.5 * n_rows)
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create GridSpec for the main grid and labels
    gs = GridSpec(n_rows + 1, n_cols + 1, figure=fig)

    # Add debug prints to see what configurations we have
    print("\nDebugging configuration matching:")
    for result_entry in results:
        if result_entry["success"]:
            config = result_entry["config"]
            path = config.get("path", "")
            print(
                f"Config: var={config.get(var)}, decision_mode={config.get('decision_mode')}, path={path}"
            )

    # Create the plots
    for (row, col), pos_config in grid_positions.items():
        sweep_val = pos_config["value"]
        col_type = pos_config["type"]

        print(
            f"\nLooking for match: row={row}, col={col}, sweep_val={sweep_val}, col_type={col_type}"
        )

        # Find the matching result for this configuration
        matching_result = None
        for result_entry in results:
            if not result_entry["success"]:
                continue

            entry_config = result_entry["config"]
            entry_path = entry_config.get("path", "")

            # For prompt types (high, medium, low)
            if col_type in ["high", "medium", "low"]:
                if (
                    entry_config.get(var) == sweep_val
                    and entry_config.get("prompt_type") == col_type
                    and entry_config.get("decision_mode") == "ai"
                ):
                    matching_result = result_entry
                    print(f"  Found prompt match: {entry_path}")
                    break

            # For adaptive theta function
            elif col_type == "adaptive":
                if (
                    entry_config.get(var) == sweep_val
                    and entry_config.get("decision_mode") == "adaptive"
                    and ("theta_k_" in entry_path or "adaptive_k-" in entry_path)
                ):
                    matching_result = result_entry
                    print(f"  Found adaptive theta match: {entry_path}")
                    break

        if not matching_result:
            print(f"  No match found for {col_type} at {sweep_val}")
            # Create an empty plot if no matching result
            ax = fig.add_subplot(gs[row + 1, col + 1])
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue

        sim_results = matching_result["results"]

        # Get history data
        sheep_history = sim_results.get("sheep_history", [])
        wolf_history = sim_results.get("wolf_history", [])
        theta_history = sim_results.get("average_theta_history", [])

        # Create subplot
        ax = fig.add_subplot(gs[row + 1, col + 1])

        # Plot data if available
        steps = range(len(sheep_history))
        if sheep_history:
            ax.plot(steps, sheep_history, color="cadetblue", linewidth=1)
        if wolf_history:
            ax.plot(steps, wolf_history, color="darkred", linewidth=1)

        # Create a twin axis for theta
        if theta_history:
            ax2 = ax.twinx()

            # Create a list of x/y points where wolves exist
            valid_x = []
            valid_theta = []
            for i, wolves in enumerate(wolf_history):
                if i < len(theta_history) and wolves > 0:
                    valid_x.append(i)
                    valid_theta.append(theta_history[i])

            # Only plot theta where wolves exist
            if valid_x and valid_theta:
                ax2.plot(valid_x, valid_theta, color="darkgreen", linewidth=1)
            ax2.set_ylim(0, 1)
            ax2.axis("off")  # Hide the second y-axis

        # Set y-limit for sheep and wolves
        if fit_sheep_max_axis and "sheep_max" in entry_config:
            # Use the sheep_max from the configuration for this specific subplot
            subplot_sheep_max = entry_config.get("sheep_max")
            ax.set_ylim(0, subplot_sheep_max * 1.1)  # Add 10% margin
        else:
            # Use the global maximum sheep population
            ax.set_ylim(0, sheep_max * 1.1)  # Add 10% margin

        # Remove ticks and labels for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # Add row labels (sweep variable values)
    for i, val in enumerate(unique_values):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.text(0.5, 0.5, f"{var}={val}", ha="center", va="center", rotation=90)
        ax.axis("off")

    # Add column labels (prompt types and adaptive theta)
    this_k = preset.get("fixed_parameters", {}).get("k")
    column_labels = [
        "Prompt: High Info",
        "Prompt: Medium Info",
        "Prompt: Low Info",
        f"Adaptive θ (k={this_k})",
    ]

    for j, label in enumerate(column_labels):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.text(0.5, 0.5, label, ha="center", va="center")
        ax.axis("off")

    # Add a title
    plt.suptitle(f"Prompt Sweep: {preset.get('preset_name', 'Unnamed')}")

    # Add a small legend in the top-left corner
    legend_ax = fig.add_subplot(gs[0, 0])
    legend_ax.plot([], [], color="cadetblue", label="Sheep")
    legend_ax.plot([], [], color="darkred", label="Wolves")
    legend_ax.plot([], [], color="darkgreen", label="Theta")
    legend_ax.legend(loc="center", frameon=False, fontsize="small")
    legend_ax.axis("off")

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the title

    # Save the figure
    output_path = os.path.join(output_dir, "prompt_sweep_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run wolf-sheep simulations using preset configurations"
    )

    # Add preset argument (required)
    parser.add_argument(
        "preset", type=str, help="Name of the preset configuration to use"
    )

    # Add optional arguments
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers",  # This default is acceptable
    )

    parser.add_argument(
        "--presets-file",
        type=str,
        default="presets.json",
        help="Path to the presets JSON file (relative to project root)",
    )

    args = parser.parse_args()

    # Load presets
    presets = load_presets(args.presets_file)

    # Check if the specified preset exists
    if args.preset not in presets:
        print(
            f"Error: Preset '{args.preset}' not found. Available presets: {', '.join(presets.keys())}"
        )
        return

    preset = presets[args.preset]
    preset_type = preset.get("preset_type")

    # Generate configurations based on preset type
    if preset_type == "single":
        # Run a single simulation with the preset's fixed parameters
        configs = [preset.get("fixed_parameters", {})]
        print(f"Running single simulation with preset '{args.preset}'")

    elif preset_type == "sweep":
        # Generate configurations for parameter sweep
        configs = generate_sweep_configs(preset)
        print(
            f"Running parameter sweep with preset '{args.preset}' ({len(configs)} configurations)"
        )

    elif preset_type == "prompt-sweep":
        # Generate configurations for prompt sweep
        configs = generate_prompt_sweep_configs(preset)
        print(
            f"Running prompt sweep with preset '{args.preset}' ({len(configs)} configurations)"
        )

    else:
        print(f"Error: Unknown preset type '{preset_type}'")
        return

    # Print experiment summary
    print(
        f"Running {len(configs)} simulations with {args.max_workers} parallel workers"
    )

    # Run simulations in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = []

        # Submit jobs
        for config in configs:
            futures.append(executor.submit(run_simulation, config))

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                success, config, sim_results = future.result()
                results.append(
                    {
                        "success": success,
                        "config": config,
                        "results": sim_results if success else None,
                    }
                )
                if success:
                    print(f"✅ Simulation completed: {config.get('path', 'unknown')}")
                else:
                    print(f"❌ Simulation failed: {config.get('path', 'unknown')}")
            except Exception as e:
                print(f"Exception occurred: {e}")

    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nExperiment complete: {successful}/{len(configs)} simulations successful")

    # Save experiment configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine where to save the experiment summary
    # Use the path from the preset if available, but avoid path doubling
    if preset_type == "single" and "path" in preset.get("fixed_parameters", {}):
        path = preset["fixed_parameters"]["path"]
        # Remove "data/results/" prefix if it exists to avoid path doubling
        if path.startswith("data/results/"):
            output_dir = os.path.join(get_project_root(), path)
        else:
            output_dir = os.path.join(get_project_root(), "data", "results", path)
    elif preset_type == "sweep" and "path" in preset.get("fixed_parameters", {}):
        path = preset["fixed_parameters"]["path"]
        # Remove "data/results/" prefix if it exists to avoid path doubling
        if path.startswith("data/results/"):
            output_dir = os.path.join(get_project_root(), path)
        else:
            output_dir = os.path.join(get_project_root(), "data", "results", path)
    elif preset_type == "prompt-sweep" and "path" in preset.get("fixed_parameters", {}):
        path = preset["fixed_parameters"]["path"]
        # Remove "data/results/" prefix if it exists to avoid path doubling
        if path.startswith("data/results/"):
            output_dir = os.path.join(get_project_root(), path)
        else:
            output_dir = os.path.join(get_project_root(), "data", "results", path)
    else:
        # Default to a results directory in the project root
        output_dir = os.path.join(get_project_root(), "data", "results", args.preset)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract key statistics for the summary
    sweep_stats = []
    for result_entry in results:
        if not result_entry["success"]:
            continue

        config = result_entry["config"]
        sim_results = result_entry["results"]

        # Create a stats entry for this configuration
        stats_entry = {
            "config": {
                var: config.get(var) for var in preset.get("sweep_variables", [])
            },
            "final_sheep": round4(sim_results.get("final_sheep")),
            "final_wolves": sim_results.get("final_wolves"),
            "last_wolf_death_step": sim_results.get(
                "last_wolf_death_step", "N/A"
            ),  # Acceptable default, N/A for null
        }

        # Add any other key metrics you want to track
        sweep_stats.append(stats_entry)

    # Sort the stats by the sweep variables for easier analysis
    if sweep_stats and "config" in sweep_stats[0] and preset.get("sweep_variables"):
        # Sort by the first sweep variable, then by the second if present
        sweep_vars = preset.get("sweep_variables", [])
        if len(sweep_vars) == 1:
            sweep_stats.sort(key=lambda x: x["config"].get(sweep_vars[0]))
        elif len(sweep_vars) >= 2:
            sweep_stats.sort(
                key=lambda x: (
                    x["config"].get(sweep_vars[0]),
                    x["config"].get(sweep_vars[1]),
                )
            )

    # Create experiment summary
    experiment_config = {
        "timestamp": timestamp,
        "preset_name": args.preset,
        "preset_version": preset.get("preset_version"),
        "preset_type": preset_type,
        "preset_description": preset.get("preset_description"),
        "num_configurations": len(configs),
        "successful_runs": successful,
        "sweep_statistics": sweep_stats,
        "results": results,
    }

    # Create visualization based on preset type
    if preset_type == "sweep" and successful > 0:
        print("Creating parameter sweep visualization...")
        viz_path = create_sweep_visualization(sweep_stats, results, preset, output_dir)
        if viz_path:
            print(f"Visualization saved to {viz_path}")
            # Add the visualization path to the experiment config
            experiment_config["visualization_path"] = str(viz_path)

    elif preset_type == "prompt-sweep" and successful > 0:
        print("Creating prompt sweep visualization...")
        viz_path = create_prompt_sweep_visualization(
            sweep_stats, results, preset, output_dir
        )
        if viz_path:
            print(f"Visualization saved to {viz_path}")
            # Add the visualization path to the experiment config
            experiment_config["visualization_path"] = str(viz_path)

    # Save experiment configuration
    config_path = Path(output_dir) / f"experiment_summary_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(experiment_config, f, indent=2)

    print(f"Experiment summary saved to {config_path}")


if __name__ == "__main__":
    main()
