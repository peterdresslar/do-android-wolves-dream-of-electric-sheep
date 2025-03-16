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
from model.utils.simulation_utils import round4, save_simulation_results

MODEL_PARAMS = {  # Model parameters have an effect on the outcome of the simulation, and thus should NEVER have default values.
    "decision_mode",  # "ai", "adaptive", "constant"
    "alpha",
    "beta",
    "gamma",
    "delta",
    "theta_start",  # Used in all decision modes.
    "randomize_theta_start",  # True or False
    "s_start",
    "w_start",
    "dt",
    "sheep_max",
    "eps",
    "churn_rate",
    "steps",
    "model_name",
    "temperature",
    "prompt_type",
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
        # Note that Model Opts can have default values
        self.opts["save_results"] = None
        self.opts["path"] = None
        self.opts["prompt_type"] = None
        self.opts["step_print"] = None
        self.opts["threads"] = None

    def create_run(self) -> ModelRun:
        """
        Create a simulation run for this model.
        """
        return ModelRun(self)


def initialize_model(**kwargs) -> Model:
    """
    Initialize the model with the given domain and agents.
    Additional keyword arguments become model parameters.

    IMPORTANT: All scientific parameters must be explicitly provided.
    No defaults are used for parameters that affect simulation outcomes.

    Decision modes:
    - "ai": Wolves make decisions using LLMs
    - "adaptive": Wolves use the adaptive theta formula
    - "constant": Wolves use a constant theta value
    """
    # First, check for required parameters common to all modes
    common_required = {
        "decision_mode",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "theta_start",
        "randomize_theta",
        "eps",
        "s_start",
        "w_start",
        "dt",
        "sheep_max",
        "steps",
    }

    missing_common = [param for param in common_required if param not in kwargs]
    if missing_common:
        raise ValueError(f"Missing required parameters: {', '.join(missing_common)}")

    # Check for decision mode-specific required parameters
    decision_mode = kwargs["decision_mode"]
    if decision_mode == "ai":
        if "model_name" not in kwargs:
            raise ValueError(
                "'model_name' parameter is required for decision_mode='ai'"
            )
        model_name = kwargs["model_name"]  # shouldn't be empty since we have the raise
        if "temperature" not in kwargs:
            raise ValueError(
                "'temperature' parameter is required for decision_mode='ai'"
            )
        temperature = kwargs["temperature"]
        if "prompt_type" not in kwargs:
            raise ValueError(
                "'prompt_type' parameter is required for decision_mode='ai'"
            )
        prompt_type = kwargs["prompt_type"]
        if "churn_rate" not in kwargs:
            raise ValueError(
                "'churn_rate' parameter is required for decision_mode='ai'"
            )
        churn_rate = kwargs["churn_rate"]

    elif decision_mode == "adaptive":
        if "k" not in kwargs:
            raise ValueError("'k' parameter is required for decision_mode='adaptive'")
        k = kwargs["k"]

    if decision_mode not in ["ai", "adaptive", "constant"]:
        raise ValueError(
            f"Invalid decision_mode: {decision_mode}. Must be 'ai', 'adaptive', or 'constant'"
        )

    # Extract domain parameters
    sheep_capacity = kwargs["sheep_max"]
    starting_sheep = kwargs["s_start"]
    starting_wolves = kwargs["w_start"]
    alpha = kwargs["alpha"]
    dt = kwargs["dt"]

    # Create domain
    model_domain = kwargs.pop(
        "domain",
        Domain(
            sheep_capacity=sheep_capacity,
            starting_sheep=starting_sheep,
            alpha=alpha,
            dt=dt,
        ),
    )

    # Extract agent parameters
    beta = kwargs["beta"]
    gamma = kwargs["gamma"]
    delta = kwargs["delta"]

    theta_start = kwargs["theta_start"]
    randomize_theta = kwargs["randomize_theta"]
    eps = kwargs["eps"]

    # Separate options from parameters
    opts = {
        # Options can have defaults
        "save_results": kwargs.pop("save_results", True),
        "path": kwargs.pop("path", "../data/results"),
        "step_print": kwargs.pop("step_print", False),
        "max_tokens": kwargs.pop(
            "max_tokens", 512
        ),  # Reasonable default for token limit
        "threads": kwargs.pop("threads", None),
    }

    # Create agents
    model_agents = Agents.create_agents(
        w_start=starting_wolves,
        decision_mode=decision_mode,
        # Wolves do not use alpha!
        beta=beta,
        gamma=gamma,
        delta=delta,
        theta_start=theta_start,
        randomize_theta=randomize_theta,
        eps=eps,
        k=kwargs.get("k"),  # This will be None when not in adaptive mode
        model_name=kwargs.get("model_name"),  # Will be None when not in AI mode
        temperature=kwargs.get("temperature"),  # Will be None when not in AI mode
        prompt_type=kwargs.get("prompt_type"),  # Will be None when not in AI mode
        churn_rate=kwargs.get("churn_rate"),  # Will be None when not in AI mode
        initial_step=0,
    )

    # Extract model parameters
    steps = kwargs["steps"]
    dt = kwargs["dt"]

    # Create the model
    model = Model(domain=model_domain, agents=model_agents, steps=steps, dt=dt)

    # Set options on the model
    model.opts.update(opts)

    # Add all parameters to model.params
    model.params.update(kwargs)

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
        agents.process_step_sync(domain, self.current_step)

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
            "last_wolf_death_step": agents.last_wolf_death_step,
            "runtime": runtime,
            "usage": self.usage.to_dict(),  # Add usage information
        }

        # If saving is enabled, prepare and save detailed results
        if self.model.opts.get("save_results", True):
            self._save_simulation_results(runtime)

        if opts.get("step_print"):
            print(f"Simulation completed in {round4(runtime)} seconds.")
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
            "last_wolf_death_step": self.model.agents.last_wolf_death_step,
            "model_params": self.model.params,
            "model_opts": self.model.opts,
            "domain": {
                "sheep_capacity": self.model.domain.sheep_capacity,
                "sheep_state": self.model.domain.sheep_state,
            },
            "agents": self.model.agents.get_agents_summary(),
            "usage": self.usage.to_dict(),
            "threads": self.model.opts.get("threads", None),
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
