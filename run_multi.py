import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path
import random
import time


def run_simulation(config):
    """Run a single simulation with the given configuration."""
    cmd_args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd_args.append(f"--{key.replace('_', '-')}")
        else:
            cmd_args.append(f"--{key.replace('_', '-')}")
            cmd_args.append(str(value))

    cmd = [sys.executable, "main.py"] + cmd_args
    print(f"Running command: {' '.join(cmd)}")

    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.returncode != 0:
        print(f"Error running simulation: {process.stderr}")
        return False, config

    print(f"Simulation completed: {config.get('model_name', 'unnamed')}")
    return True, config


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple wolf-sheep simulations in parallel"
    )

    # Basic experiment configuration
    parser.add_argument(
        "--count", type=int, default=4, help="Number of simulations to run"
    )
    parser.add_argument(
        "--max-workers", type=int, default=2, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--base-name", type=str, default="Model", help="Base name for the models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/results",
        help="Directory to save results",
    )

    # Parameter ranges
    parser.add_argument(
        "--steps-range",
        type=str,
        default="250",
        help="Range of steps (single value or min,max)",
    )
    parser.add_argument(
        "--sheep-range",
        type=str,
        default="100",
        help="Range of initial sheep (single value or min,max)",
    )
    parser.add_argument(
        "--wolves-range",
        type=str,
        default="10",
        help="Range of initial wolves (single value or min,max)",
    )
    parser.add_argument(
        "--sheep-max-range",
        type=str,
        default="110",
        help="Range of max sheep (single value or min,max)",
    )

    # Model parameters ranges
    parser.add_argument(
        "--alpha-range",
        type=str,
        default="1.0",
        help="Range of alpha values (single value or min,max)",
    )
    parser.add_argument(
        "--beta-range",
        type=str,
        default="0.1",
        help="Range of beta values (single value or min,max)",
    )
    parser.add_argument(
        "--gamma-range",
        type=str,
        default="1.5",
        help="Range of gamma values (single value or min,max)",
    )
    parser.add_argument(
        "--delta-range",
        type=str,
        default="0.75",
        help="Range of delta values (single value or min,max)",
    )

    # AI options
    parser.add_argument(
        "--prompt-types",
        type=str,
        default="high",
        help="Comma-separated list of prompt types to use (high,low)",
    )
    parser.add_argument(
        "--include-no-ai", action="store_true", help="Include runs with AI disabled"
    )
    parser.add_argument(
        "--churn-rate-range",
        type=str,
        default="0.05",
        help="Range of churn rates (single value or min,max)",
    )

    # Experiment design
    parser.add_argument(
        "--random-seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Delay in seconds between starting simulations",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.random_seed is not None:
        random.seed(args.random_seed)

    # Parse ranges
    def parse_range(range_str):
        if "," in range_str:
            min_val, max_val = map(float, range_str.split(","))
            return min_val, max_val
        else:
            val = float(range_str)
            return val, val

    steps_min, steps_max = parse_range(args.steps_range)
    sheep_min, sheep_max = parse_range(args.sheep_range)
    wolves_min, wolves_max = parse_range(args.wolves_range)
    sheep_max_min, sheep_max_max = parse_range(args.sheep_max_range)
    alpha_min, alpha_max = parse_range(args.alpha_range)
    beta_min, beta_max = parse_range(args.beta_range)
    gamma_min, gamma_max = parse_range(args.gamma_range)
    delta_min, delta_max = parse_range(args.delta_range)
    churn_min, churn_max = parse_range(args.churn_rate_range)

    # Parse prompt types
    prompt_types = args.prompt_types.split(",")

    # Generate simulation configurations
    configs = []

    for i in range(args.count):
        # Generate random values within ranges
        steps = int(random.uniform(steps_min, steps_max))
        sheep = int(random.uniform(sheep_min, sheep_max))
        wolves = int(random.uniform(wolves_min, wolves_max))
        sheep_max = int(random.uniform(sheep_max_min, sheep_max_max))
        alpha = random.uniform(alpha_min, alpha_max)
        beta = random.uniform(beta_min, beta_max)
        gamma = random.uniform(gamma_min, gamma_max)
        delta = random.uniform(delta_min, delta_max)
        churn_rate = random.uniform(churn_min, churn_max)

        # For each configuration, decide if it's AI or no-AI
        use_ai = True
        if args.include_no_ai and random.random() < 0.5:
            use_ai = False

        # Select a prompt type if using AI
        prompt_type = random.choice(prompt_types) if use_ai else None

        config = {
            "model_name": f"{args.base_name}_{i+1}",
            "steps": steps,
            "sheep": sheep,
            "wolves": wolves,
            "sheep_max": sheep_max,
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "gamma": round(gamma, 3),
            "delta": round(delta, 3),
            "churn_rate": round(churn_rate, 3),
            "output_dir": args.output_dir,
        }

        if not use_ai:
            config["no_ai"] = True
            config["theta"] = 0.5  # Default theta for no-AI runs
        elif prompt_type:
            config["prompt_type"] = prompt_type

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

        # Submit jobs with optional delay
        for config in configs:
            futures.append(executor.submit(run_simulation, config))
            if args.delay > 0:
                time.sleep(args.delay)

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                success, config = future.result()
                results.append({"success": success, "config": config})
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
    import json
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(
            {"args": vars(args), "configs": configs, "results": results}, f, indent=2
        )

    print(f"Experiment configuration saved to {experiment_dir}")


if __name__ == "__main__":
    main()
