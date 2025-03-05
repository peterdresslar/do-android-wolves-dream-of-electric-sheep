# model.py
from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from typing import Any

from model.agents import Agents
from model.domain import Domain
from model.utils.data_types import Usage, set_current_usage
from model.utils.init_utils import initialize_utils
from model.utils.simulation_utils import save_simulation_results

MODEL_PARAMS = {
    "alpha": None,
    "beta": None,
    "gamma": None,
    "delta": None,
    "theta_star": None,
    "s_start": None,
    "w_start": None,
    "dt": None,
    "sheep_max": None,
    "eps": None,
    "steps": None,
}


@dataclass
class Model:
    """
    Core model class that manages domain and agents.

    Supports:
      - Explicit parameters: steps and dt.
      - Additional keyword arguments (captured in params).
      - A state dictionary for simulation state.
    """

    # TODO: correctly formalize the existence and operation of time for this local cosmos

    domain: Domain
    agents: Agents
    steps: int = None
    dt: float = None
    t: int = field(init=False)
    params: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict, init=False)
    opts: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.t = (
            int(self.steps * self.dt)
            if self.steps is not None and self.dt is not None
            else None
        )
        self.opts["no_ai"] = None
        self.opts["model_name"] = None
        self.opts["models"] = []
        self.opts["churn_rate"] = None
        self.opts["save_results"] = None
        self.opts["path"] = None
        self.opts["prompt_type"] = None
        self.opts["step_print"] = None

    def create_run(self) -> ModelRun:
        """
        Create a simulation run for this model.
        """
        return ModelRun(self)


def initialize_model(**kwargs) -> Model:
    """
    Initialize the model with the given domain and agents.
    Additional keyword arguments become model parameters.

    This function merges defaults from config.py with any overrides
    provided in kwargs so that keyword arguments can override the config.
    """
    defaults = MODEL_PARAMS.copy()
    defaults.update(kwargs)

    # Extract domain and agent parameters
    sheep_capacity = defaults.get("sheep_max")
    starting_sheep = defaults.get("s_start")
    starting_wolves = defaults.get(
        "w_start", 10
    )  # Add default value for starting_wolves

    # Create domain and agents
    model_domain = defaults.pop(
        "domain",
        Domain(
            sheep_capacity=sheep_capacity,
            starting_sheep=starting_sheep,
        ),
    )

    # Extract agent parameters
    beta = defaults.get("beta")
    gamma = defaults.get("gamma")
    delta = defaults.get("delta")

    # Set initial theta value based on theta_star or default
    if defaults.get("no_ai") and defaults.get("theta_star") is not None:
        # If no_ai is True and theta_star is provided, use it as initial theta
        theta = defaults.get("theta_star")
    else:
        # Otherwise, use 0.5 as a default initial theta
        theta = 0.5

    # Create opts dictionary
    opts = {
        "no_ai": defaults.get("no_ai"),
        "churn_rate": defaults.get("churn_rate"),
        "save_results": defaults.get("save_results"),
        "path": defaults.get("path"),
        "prompt_type": defaults.get("prompt_type"),
        "model_name": defaults.get("model_name"),
        "step_print": defaults.get("step_print"),
    }

    # Create agents with cleaner parameter passing
    model_agents = Agents.create_agents(
        n_wolves=starting_wolves,
        beta=beta,
        gamma=gamma,
        delta=delta,
        theta=theta,
        opts=opts,
        initial_step=0,  # Start at step 0
    )

    # Extract model parameters
    steps = defaults.get("steps")
    dt = defaults.get("dt")

    # Create the model
    model = Model(domain=model_domain, agents=model_agents, steps=steps, dt=dt)

    # Set options on the model too
    model.opts.update(opts)

    # Add remaining parameters to model.params
    model.params.update(defaults)

    return model


class ModelRun:
    """
    Represents a simulation run of the Model.
    This class handles the simulation loop, yielding intermediate states and
    final results.

    Since we are working with AIs, we track usage for each ModelRun instance.

    """

    def __init__(self, model: Model):
        self.model = model
        self.current_step = 0
        self.snapshots = []
        self.usage = Usage()

    def step(self) -> dict[str, Any]:
        """
        Execute one simulation step, updating the domain's state by interacting with agents.
        """
        # Get current parameters and domain
        params = self.model.params  # shouldnʻt change
        domain = self.model.domain
        agents = self.model.agents

        # 1. Reset Domain for new step
        domain.reset_accumulators()

        # 2. Process wolves
        agents.process_step_sync(params, domain, self.current_step)

        # 3. Process wolf effects on the domain (apply accumulated changes)
        # Note that in the next two steps
        # we are effectively managing partial time across domain and agents
        net_wolves_change = domain.accumulate_and_fit(params)

        if params.get("step_print"):
            print(
                f"Step {self.current_step}: net_wolves_change: {net_wolves_change}: avg theta: {agents.get_mean_theta()}"
            )

        # 4. Handle wolf population changes (moved to Agents class)
        agents.handle_population_changes(net_wolves_change, self.current_step)

        # 5. Process sheep growth
        domain.process_sheep_growth(params)

        snapshot = {
            "step": self.current_step,
            "sheep": domain.sheep_state,
            "wolves": agents.living_wolves_count,
            "thetas": agents.get_current_thetas(),
            "mean_theta": agents.get_mean_theta(),
        }

        return snapshot

    def run(self) -> dict[str, Any]:
        """
        Run the simulation to completion and return a simplified results object.

        This method focuses on running the simulation and returning essential data
        needed by notebook functions or other callers.
        """

        set_current_usage(self.usage)
        start_time = time.time()
        params = self.model.params
        opts = self.model.opts
        agents = self.model.agents
        domain = self.model.domain

        if opts.get("step_print"):
            print(
                f"Starting simulation at {datetime.datetime.fromtimestamp(start_time)} with {self.model.steps} steps."
            )
            print(f"Model params: {params}")
            print(f"Model opts: {opts}")
            print(f"Domain starting sheep: {domain.sheep_state}")
            print(
                f"Agents starting wolves: {len([w for w in agents.wolves if w.alive])}"
            )

        i = 0  # dont increment on first step, allowing start at zero
        for _ in range(self.model.steps):
            self.current_step += i
            snapshot = self.step()
            self.snapshots.append(snapshot)
            i = 1  # i learned this approach in BASIC

        end_time = time.time()
        runtime = end_time - start_time

        # Prepare a simplified results object for return
        results = {
            "steps": self.current_step,
            "sheep_history": domain.sheep_history,
            "wolf_history": agents.get_living_wolf_count_history(),
            "average_theta_history": agents.get_average_theta_history(),
            "final_sheep": domain.sheep_state,
            "final_wolves": agents.living_wolves_count,
            "runtime": runtime,
            "usage": self.usage.to_dict(),  # Add usage information
        }

        # If saving is enabled, prepare and save detailed results
        if self.model.opts.get("save_results", True):
            self._save_simulation_results(runtime)

        if opts.get("step_print"):
            print(f"Simulation completed in {runtime} seconds.")
            print(f"Usage: {self.usage.to_dict()}")

        return results

    def _prepare_detailed_results(self, runtime: float) -> dict[str, Any]:
        """
        Prepare a detailed results object suitable for saving to files.
        """
        # Ensure all histories have the same length
        sheep_history = self.model.domain.sheep_history
        wolf_history = self.model.agents.get_living_wolf_count_history()
        theta_history = self.model.agents.average_thetas

        # Find the maximum length
        max_length = max(len(sheep_history), len(wolf_history), len(theta_history))

        # Pad arrays if needed
        if len(sheep_history) < max_length:
            last_value = sheep_history[-1] if sheep_history else 0
            sheep_history = sheep_history + [last_value] * (
                max_length - len(sheep_history)
            )

        if len(wolf_history) < max_length:
            last_value = wolf_history[-1] if wolf_history else 0
            wolf_history = wolf_history + [last_value] * (
                max_length - len(wolf_history)
            )

        if len(theta_history) < max_length:
            last_value = theta_history[-1] if theta_history else 0
            theta_history = theta_history + [last_value] * (
                max_length - len(theta_history)
            )

        # Get the model_name from model_names if available
        model_name = self.model.params.get("model_name", "Model")
        if (
            "model_names" in self.model.params
            and isinstance(self.model.params["model_names"], list)
            and self.model.params["model_names"]
        ):
            model_name = self.model.params["model_names"][0]

        detailed_results = {
            "runtime": runtime,
            "model_name": model_name,
            "prompt_type": self.model.opts.get("prompt_type", "high"),
            "steps": self.current_step,
            "sheep_history": sheep_history,
            "wolf_history": wolf_history,
            "average_theta_history": theta_history,
            "final_sheep": self.model.domain.sheep_state,
            "final_wolves": self.model.agents.living_wolves_count,
            "model_params": self.model.params,
            "model_opts": self.model.opts,
            "domain": {
                "sheep_capacity": self.model.domain.sheep_capacity,
                "sheep_state": self.model.domain.sheep_state,
            },
            "agents": self.model.agents.get_agents_summary(),
            "usage": self.usage.to_dict(),
        }

        return detailed_results

    def _save_simulation_results(self, runtime: float) -> None:
        """
        Prepare detailed results and save them using the simulation_utils function.
        """
        detailed_results = self._prepare_detailed_results(runtime)
        results_path = self.model.opts.get("path", "../data/results")

        # Save results to file
        save_simulation_results(detailed_results, results_path)


def run(**kwargs) -> dict[str, Any]:
    """
    Run the model synchronously and return the results.
    This will fail in contexts that already have an event loop running.
    """
    model = initialize_model(**kwargs)
    # Initialize the utils environment
    initialize_utils()
    runner = model.create_run()
    return runner.run()
