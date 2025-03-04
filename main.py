# main.py
import argparse
import json
import sys
from pathlib import Path

from model.model import run
from model.utils.utils import VALID_MODELS

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def get_preset(preset_name: str) -> dict:
    """
    Get a preset from the preset.json file.
    """
    with open("presets.json") as f:
        preset = json.load(f)
        if preset_name not in preset:
            raise ValueError(f"Preset {preset_name} not found in preset.json")
        return preset[preset_name]


def handle_preset(params: dict) -> dict:
    """
    Handle the preset for the simulation.
    Append all preset values to the params dict.
    Command-line arguments override preset values.
    """
    preset = get_preset(params["preset"])

    # Start with a clean dictionary
    result_params = {}

    # First, add all preset values
    for key, value in preset.items():
        result_params[key] = value

    # Then override with any explicitly provided command-line arguments
    # Only include non-None values from params to preserve preset values
    for key, value in params.items():
        if key != "preset" and value is not None:
            result_params[key] = value

    return result_params


def main():
    parser = argparse.ArgumentParser(
        description="Run wolf-sheep predator-prey simulation"
    )

    parser.add_argument(
        "--preset", type=str, default="base", help="Preset to use for the simulation"
    )

    # Basic simulation parameters
    parser.add_argument(
        "--steps", type=int, default=250, help="Number of simulation steps"
    )
    parser.add_argument(
        "--sheep", type=int, default=100, help="Initial sheep population"
    )
    parser.add_argument(
        "--wolves", type=int, default=10, help="Initial wolf population"
    )
    parser.add_argument(
        "--sheep-max", type=int, default=110, help="Maximum sheep capacity"
    )

    # Model parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Sheep growth rate")
    parser.add_argument("--beta", type=float, default=0.1, help="Predation rate")
    parser.add_argument("--gamma", type=float, default=1.5, help="Wolf death rate")
    parser.add_argument(
        "--delta", type=float, default=0.75, help="Conversion efficiency"
    )

    # AI and execution options
    parser.add_argument("--no-ai", action="store_true", help="Disable AI for wolves")
    parser.add_argument(
        "--theta-star",
        type=float,
        default=None,
        help="Fixed theta value to use as a constant (if provided)"
    )
    parser.add_argument(
        "--churn-rate",
        type=float,
        default=0.05,
        help="Rate at which wolves update their decisions",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model-name", type=str, default="Model", help="Name prefix for the model run"
    )
    # or the user can pass a list of model names
    parser.add_argument(
        "--model-names",
        type=str,
        default=VALID_MODELS,
        help="List of model names to run",
    )

    # Add this to the argument parser section, after the AI options
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["high", "medium", "low"],
        default="high",
        help="Type of prompt to use for AI wolves: high or low information",
    )

    parser.add_argument(
        "--no-save", action="store_true", help="Don't save simulation results"
    )

    parser.add_argument(
        "--step-print", action="store_true", help="Print step information"
    )

    args = parser.parse_args()

    # Prepare parameters for the simulation
    # Keep all the default values from argparse
    params = {
        "preset": args.preset,
        "model_name": args.model_name,
        "model_names": args.model_names,
        "steps": args.steps,
        "s_start": args.sheep,
        "w_start": args.wolves,
        "sheep_max": args.sheep_max,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta": args.delta,
        "no_ai": args.no_ai,
        "theta_star": args.theta_star,
        "churn_rate": args.churn_rate,
        "save_results": not args.no_save,
        "path": args.output_dir,
        "prompt_type": args.prompt_type,
        "step_print": args.step_print,
    }

    # Apply preset, allowing command-line args to override
    params = handle_preset(params)

    print(f"Running simulation with parameters: {params}")

    # Run the simulation
    results = run(**params)

    # Find the most recent results directory

    print("Simulation completed successfully!")
    print(f"Final sheep: {results['final_sheep']}")
    print(f"Final wolves: {results['final_wolves']}")

    return results


if __name__ == "__main__":
    main()
