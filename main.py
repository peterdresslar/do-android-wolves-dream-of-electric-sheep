# main.py
import argparse
import os
import sys
from pathlib import Path

from model.model import run
from model.utils import VALID_MODELS

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run wolf-sheep predator-prey simulation"
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
        "--theta", type=float, default=0.5, help="Fixed theta value when AI is disabled"
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
        choices=["high", "low"],
        default="high",
        help="Type of prompt to use for AI wolves: high or low information",
    )

    parser.add_argument(
        "--no-save", action="store_true", help="Don't save simulation results"
    )

    args = parser.parse_args()

    # Prepare parameters for the simulation
    params = {
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
        "theta_star": args.theta,
        "churn_rate": args.churn_rate,
        "save_results": not args.no_save,
        "path": args.output_dir,
        "prompt_type": args.prompt_type,
    }

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
