#!/usr/bin/env python3
# run_sheep_experiment.py
import argparse
import concurrent.futures
import datetime
import json
import os
import time
from pathlib import Path

# Import the run function directly from model.py
from model.model import run

# Constants
STEPS = 500
MODEL_NAME = "gpt-4o-mini"
OUTPUT_DIR = "data/results/sheep_experiment"


def run_simulation(config):
    """Run a single simulation with the given configuration."""
    print(f"Running simulation with config: {config}")

    try:
        # Call the run function directly
        start_time = time.time()
        results = run(**config)
        end_time = time.time()

        print(
            f"Simulation completed in {end_time - start_time:.2f} seconds: {config['model_name']}"
        )

        # Add runtime to results
        results["runtime"] = end_time - start_time
        return True, config, results
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        return False, config, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Run wolf-sheep simulations with varying sheep counts and prompt types"
    )

    # Model Name
    parser.add_argument(
        "--model-name", type=str, default=MODEL_NAME, help="Model name"
    )

    # Basic experiment configuration
    parser.add_argument(
        "--max-workers", type=int, default=3, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--steps", type=int, default=STEPS, help="Number of simulation steps"
    )
    parser.add_argument(
        "--wolves", type=int, default=10, help="Initial wolf population"
    )
    parser.add_argument(
        "--sheep-max", type=int, default=1000, help="Maximum sheep capacity"
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Time step size")

    # Model parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Sheep growth rate")
    parser.add_argument("--beta", type=float, default=0.1, help="Predation rate")
    parser.add_argument("--gamma", type=float, default=1.5, help="Wolf death rate")
    parser.add_argument(
        "--delta", type=float, default=0.75, help="Conversion efficiency"
    )
    parser.add_argument("--churn-rate", type=float, default=0.05, help="Churn rate")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define sheep counts to test
    sheep_counts = [19, 18, 17, 16, 15]

    # Define prompt types
    prompt_types = ["high", "medium", "low"]

    # Generate simulation configurations
    configs = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for sheep_count in sheep_counts:
        for prompt_type in prompt_types:
            config = {
                "model_name": args.model_name,
                "steps": args.steps,
                "s_start": sheep_count,  # Use s_start instead of sheep
                "w_start": args.wolves,  # Use w_start instead of wolves
                "sheep_max": args.sheep_max,
                "dt": args.dt,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "delta": args.delta,
                "churn_rate": args.churn_rate,
                "prompt_type": prompt_type,
                "path": args.output_dir,  # Use path instead of output_dir
                "save_results": True,
                "step_print": True,
            }
            configs.append(config)

    # Print experiment summary
    print(
        f"Running {len(configs)} simulations with {args.max_workers} parallel workers"
    )
    for i, config in enumerate(configs):
        print(f"Simulation {i+1}: {config}")

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
                    print(f"✅ Simulation completed: {config.get('model_name')}")
                else:
                    print(f"❌ Simulation failed: {config.get('model_name')}")
            except Exception as e:
                print(f"Exception occurred: {e}")

    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nExperiment complete: {successful}/{len(configs)} simulations successful")

    # Save experiment configuration
    experiment_config = {
        "timestamp": timestamp,
        "sheep_counts": sheep_counts,
        "prompt_types": prompt_types,
        "wolves": args.wolves,
        "sheep_max": args.sheep_max,
        "steps": args.steps,
        "dt": args.dt,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta": args.delta,
        "churn_rate": args.churn_rate,
        "results": results,
    }

    config_path = Path(args.output_dir) / f"experiment_config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(experiment_config, f, indent=2)

    print(f"Experiment configuration saved to {config_path}")


if __name__ == "__main__":
    main()
