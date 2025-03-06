#!/usr/bin/env python3
# run_multi_experiment.py
import argparse
import concurrent.futures
import datetime
import json
import os
import time
from pathlib import Path

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

        with open(presets_path, "r") as f:
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


def run_simulation(config):
    """Run a single simulation with the given configuration."""
    print(f"Running simulation with config: {config}")

    try:
        # Call the run function directly
        start_time = time.time()
        results = run(**config)
        end_time = time.time()

        print(
            f"Simulation completed in {end_time - start_time:.2f} seconds: {config.get('path', 'unknown')}"
        )

        # Add runtime to results
        results["runtime"] = end_time - start_time
        return True, config, results
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        return False, config, {"error": str(e)}


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
        "--max-workers", type=int, default=3, help="Maximum number of parallel workers"
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
            "config": {var: config.get(var) for var in preset.get("sweep_variables", [])},
            "final_sheep": round4(sim_results.get("final_sheep", 0)),
            "final_wolves": sim_results.get("final_wolves", 0),
            "step_of_last_wolf_death": sim_results.get("step_of_last_wolf_death", "N/A")
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
            sweep_stats.sort(key=lambda x: (
                x["config"].get(sweep_vars[0]), 
                x["config"].get(sweep_vars[1])
            ))

    # Create experiment summary
    experiment_config = {
        "timestamp": timestamp,
        "preset_name": args.preset,
        "preset_type": preset_type,
        "preset_description": preset.get("preset_description", ""),
        "num_configurations": len(configs),
        "successful_runs": successful,
        "sweep_statistics": sweep_stats,
        "results": results,
    }

    config_path = Path(output_dir) / f"experiment_summary_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(experiment_config, f, indent=2)

    print(f"Experiment summary saved to {config_path}")


if __name__ == "__main__":
    main()
