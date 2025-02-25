# model.py
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from model.agents import Agents
from model.domain import Domain
from model.simulation_utils import save_simulation_results

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
    "s_max": 110,  # an upper bound on sheep, for clarity
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
        self.opts["path"] = "/data/results"
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

    # Extract domain parameters
    sheep_capacity = defaults.get("s_max", 110)
    starting_sheep = defaults.get("s_start", 100)
    starting_wolves = defaults.get("w_start", 10)

    # Create domain and agents
    model_domain = defaults.pop(
        "domain",
        Domain(
            sheep_capacity=sheep_capacity,
            starting_sheep=starting_sheep,
            starting_wolves=starting_wolves,
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
        "path": defaults.get("path", "/data/results"),
    }

    # Create agents with cleaner parameter passing
    model_agents = Agents(
        n_wolves=model_domain.starting_wolves,
        beta=beta,
        gamma=gamma,
        delta=delta,
        theta=theta,
        opts=opts,
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
    """

    def __init__(self, model: Model):
        self.model = model
        self.current_step = 0
        self.history: list[dict[str, Any]] = []

        # Store wolf and sheep counts for history
        self.sheep_history = [model.domain.s_state]
        self.wolf_history = [len([w for w in model.agents.wolves if w.alive])]

        # Initialize theta_history with the initial thetas of all wolves
        # For no_ai mode, this will be the constant theta value
        living_wolves = [w for w in model.agents.wolves if w.alive]

        # If no wolves are alive, use an empty list
        # Otherwise, collect the thetas of all living wolves
        if not living_wolves:
            initial_thetas = []
        else:
            # For newly created wolves, thetas will be empty
            # We need to make sure they have a theta value before we can collect them

            # If no_ai is true, use the theta from model params
            if model.opts.get("no_ai", False):
                theta = model.params.get("theta", 0.5)
                # Make sure all wolves have this theta in their list
                for wolf in living_wolves:
                    if not wolf.thetas:
                        wolf.thetas.append(theta)
                initial_thetas = [wolf.thetas[-1] for wolf in living_wolves]
            else:
                # For AI mode, we'll initialize with default values
                # This will be updated in the first step
                initial_thetas = [1.0 for _ in living_wolves]

        self.theta_history = [initial_thetas]

    async def step(self) -> dict[str, Any]:
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

        # 2. Process wolf effects on the domain (apply accumulated changes)
        net_wolves_change = domain.accumulate_and_fit(
            params
        )  # this is a slightly different approach than the reference notebook

        # 3. Handle wolf population changes # TODO definitely should not be here
        if net_wolves_change > 0:
            agents.birth_wolves(self.current_step, net_wolves_change)
        elif net_wolves_change < 0:
            agents.kill_wolves(self.current_step, abs(net_wolves_change))

        # 4. Process sheep growth
        # Note that in the reference notebook, this is process_s_euler_forward()
        domain.process_sheep_growth(params)

        # 5. Increment step counter
        domain.increment_step()

        # Capture a snapshot of state at this step
        living_wolves = [w for w in agents.wolves if w.alive]
        current_thetas = [w.thetas[-1] if w.thetas else 1.0 for w in living_wolves]

        snapshot = {
            "step": self.current_step,
            "sheep": domain.s_state,
            "wolves": len(living_wolves),
            "thetas": current_thetas,
            "mean_theta": (
                sum(current_thetas) / len(current_thetas) if current_thetas else 0
            ),
        }

        # Store history
        self.history.append(snapshot)
        self.sheep_history.append(domain.s_state)
        self.wolf_history.append(len(living_wolves))
        self.theta_history.append(current_thetas)

        self.current_step += 1

        # Allow other tasks to run (simulate asynchronous delay)
        await asyncio.sleep(0)

        return snapshot

    async def run(self) -> dict[str, Any]:
        """
        Run the simulation to completion and return the final state.
        """
        start_time = time.time()
        print(f"Starting simulation at {start_time} with {self.model.steps} steps.")
        print(f"Model params: {self.model.params}")
        print(f"Model opts: {self.model.opts}")

        while self.current_step < self.model.steps:
            await self.step()

        end_time = time.time()
        runtime = end_time - start_time

        # Prepare final results
        final_results = {
            "steps": self.current_step,
            "sheep_history": self.sheep_history,
            "wolf_history": self.wolf_history,
            "theta_history": self.theta_history,
            "average_theta_history": self.model.agents.average_thetas,
            "final_sheep": self.model.domain.s_state,
            "final_wolves": len([w for w in self.model.agents.wolves if w.alive]),
            "history": self.history,
            "runtime": runtime
        }

        # Include detailed model and agent information if requested via opts
        if self.model.opts.get('save_results', True):
            final_results['model_params'] = self.model.params
            final_results['domain'] = {
                'sheep_capacity': self.model.domain.sheep_capacity,
                's_state': self.model.domain.s_state
            }
            final_results['agents'] = [
                {
                    'wolf_id': w.wolf_id,
                    'thetas': w.thetas,
                    'alive': w.alive,
                    'born_at_step': w.born_at_step,
                    'died_at_step': w.died_at_step,
                    'prompts': w.prompts  # new telemetry field
                } for w in self.model.agents.wolves
            ]
            # Save results to file
            save_simulation_results(final_results, self.model.opts.get('path', '/data/results'))
            print(f"Simulation completed in {runtime} seconds.")

        return final_results

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """
        Asynchronously iterate over simulation steps yielding intermediate state.
        """
        while self.current_step < self.model.steps:
            snapshot = await self.step()
            yield snapshot


def run_coroutine(**kwargs) -> asyncio.coroutine:
    """
    Return a coroutine that will run the model.
    For use with await in async contexts.
    """
    model = initialize_model(**kwargs)
    runner = model.create_run()
    return runner.run()


def run(**kwargs) -> dict[str, Any]:
    """
    Run the model synchronously and return the results.
    This will fail in contexts that already have an event loop running.
    """
    model = initialize_model(**kwargs)
    runner = model.create_run()
    return asyncio.run(runner.run())
