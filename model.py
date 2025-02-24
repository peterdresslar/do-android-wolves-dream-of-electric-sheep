# model.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agents import Agents
from domain import Domain

# Not converting sheep from the ODE for now
MODEL_PARAMS = {
    'alpha': 1,
    'beta': 0.1, # we need this at the model level only for the reference ODE
    'gamma': 1.5, # we need this at the model level only for the reference ODE
    'delta': .75, # we need this at the model level only for the reference ODE
    's_start': 100,
    'w_start': 10,
    'dt': .02,
    's_max': 110,  # an upper bound on sheep, for clarity
    'eps': 0.0001, # will give a dead sheep bounce if nonzero! set to zero if you want "real" last-wolf scenario.
    'steps': 250
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
    domain: Domain
    agents: Agents
    steps: int = 250
    dt: float = 0.02
    t: int = field(init=False)
    params: dict[str, Any] = field(default_factory=dict, init=False)
    state: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.t = int(self.steps * self.dt)
        self.params["steps"] = self.steps
        self.params["dt"] = self.dt

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

    model_domain = defaults.pop("domain", Domain(100, 10, 10))
    model_agents = defaults.pop("agents", Agents(n_wolves=model_domain.starting_wolves))
    steps = defaults.pop("steps", 250)
    dt = defaults.pop("dt", 0.02)

    model = Model(domain=model_domain, agents=model_agents, steps=steps, dt=dt)

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

    async def step(self) -> dict[str, Any]:
        """
        Execute one simulation step, updating the model's state.
        Replace this placeholder logic with your actual simulation behavior.
        """
        # Example simulation logic:
        self.model.state["step"] = self.current_step
        # Update domain state (this is an example; replace with real domain dynamics)
        self.model.domain.state = {"time": self.current_step * self.model.dt}
        # Let agents interact (assuming an update method and a state attribute on agents)
        if hasattr(self.model.agents, "update"):
            self.model.agents.update(self.model.domain.state)

        # Capture a snapshot of state at this step.
        snapshot = {
            "step": self.current_step,
            "domain_state": self.model.domain.state.copy()
                if hasattr(self.model.domain, "state") else {},
            "agents_state": self.model.agents.state.copy()
                if hasattr(self.model.agents, "state") else {}
        }
        self.history.append(snapshot)
        self.current_step += 1
        # Allow other tasks to run (simulate asynchronous delay)
        await asyncio.sleep(0)
        return snapshot

    async def run(self) -> dict[str, Any]:
        """
        Run the simulation to completion and return the final state.
        """
        while self.current_step < self.model.steps:
            await self.step()
        return self.model.state

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """
        Asynchronously iterate over simulation steps yielding intermediate state.
        """
        while self.current_step < self.model.steps:
            snapshot = await self.step()
            yield snapshot


def run(**kwargs) -> dict[str, Any]:
    """
    Convenience method to initialize and run a model synchronously.
    """
    model = initialize_model(**kwargs)
    runner = model.create_run()
    return asyncio.run(runner.run())


