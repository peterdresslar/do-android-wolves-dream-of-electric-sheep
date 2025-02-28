# model.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from model.agents import Agents
from model.domain import Domain
from model.simulation_utils import save_simulation_results

from model.utils import Usage, set_current_usage, get_current_usage

# Not converting sheep from the ODE for now
MODEL_PARAMS = {
    "alpha": 1,
    "beta": 0.1,  # we need this at the model level only for the reference ODE
    "gamma": 1.5,  # we need this at the model level only for the reference ODE
    "delta": 0.75,  # we need this at the model level only for the reference ODE
    "theta_star": 0.25,  # we might run without the AIs, in which case we could use this default
    "s_start": 100,
    "w_start": 10,
    "dt": 0.02,
    "sheep_max": 110,  # an upper bound on sheep, for clarity
    "eps": 0.0001,  # will give a dead sheep bounce if nonzero! set to zero if you want "real" last-wolf scenario.
    "steps": 250,
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
    steps: int = 250
    dt: float = 0.02
    t: int = field(init=False)
    params: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict, init=False)
    opts: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.t = int(self.steps * self.dt)
        self.opts["no_ai"] = False
        self.opts["churn_rate"] = 0.05
        self.opts["save_results"] = True
        self.opts["path"] = "../data/results"
        # If no_ai is set in opts and theta isn't already in params, add it
        if self.opts.get("no_ai", False) and "theta" not in self.params:
            self.params["theta"] = self.params.get("theta_star", 0.5)

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
    sheep_capacity = defaults.get("sheep_max", 110)
    starting_sheep = defaults.get("s_start", 100)
    starting_wolves = defaults.get("w_start", 10)

    # Create domain and agents
    model_domain = defaults.pop(
        "domain",
        Domain(
            sheep_capacity=sheep_capacity,
            starting_sheep=starting_sheep,
        ),
    )

    # Extract agent parameters
    beta = defaults.get("beta", 0.1)
    gamma = defaults.get("gamma", 1.5)
    delta = defaults.get("delta", 0.75)

    if defaults.get("no_ai", False):
        # need a theta to run without AIs
        theta = defaults.get("theta_star", 0.5)
    else:
        theta = None  # Don't pass theta when AI is enabled

    # Create opts dictionary
    opts = {
        "no_ai": defaults.get("no_ai", False),
        "churn_rate": defaults.get("churn_rate", 0.05),
        "save_results": defaults.get("save_results", True),
        "path": defaults.get("path", "../data/results"),
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
    steps = defaults.get("steps", 250)
    dt = defaults.get("dt", 0.02)

    # Create the model
    model = Model(domain=model_domain, agents=model_agents, steps=steps, dt=dt)

    # Set options on the model too
    model.opts.update(opts)

    # If no_ai is True, explicitly store theta in model params
    if opts["no_ai"]:
        model.params["theta"] = theta

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

        print(f"net_wolves_change: {net_wolves_change}")

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

        print(f"Starting simulation at {start_time} with {self.model.steps} steps.")
        print(f"Model params: {params}")
        print(f"Model opts: {opts}")
        print(f"Domain starting sheep: {domain.sheep_state}")
        print(f"Agents starting wolves: {len([w for w in agents.wolves if w.alive])}")

        i = 0 # dont increment on first step, allowing start at zero
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
            print(f"Simulation completed in {runtime} seconds.")

        print(f"Usage: {self.usage.to_dict()}")

        return results

    def _prepare_detailed_results(self, runtime: float) -> dict[str, Any]:
        """
        Prepare a detailed results object suitable for saving to files.

        This separates the data preparation from the simulation running logic.
        """

        detailed_results = {
            "runtime": runtime,
            "model_name": self.model.params.get("model_name", "model"),
            "steps": self.current_step,
            "sheep_history": self.model.domain.sheep_history,
            "wolf_history": self.model.agents.get_living_wolf_count_history(),
            "average_theta_history": self.model.agents.average_thetas,
            "final_sheep": self.model.domain.sheep_state,
            "final_wolves": self.model.agents.living_wolves_count,
            "model_params": self.model.params,
            "model_opts": self.model.opts,
            "domain": {
                "sheep_capacity": self.model.domain.sheep_capacity,
                "sheep_state": self.model.domain.sheep_state,
            },
            "agents": self.model.agents.get_agents_summary(),  # list of dict per wolf
            "usage": self.usage.to_dict(),  # Add usage information
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
    runner = model.create_run()
    return runner.run()
