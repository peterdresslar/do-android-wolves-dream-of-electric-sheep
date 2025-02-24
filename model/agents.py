# agents.py

import random
from dataclasses import dataclass, field

from .utils import get_wolf_response


@dataclass
class Wolf:
    wolf_id: int
    beta: float = 0.1  # Predation rate
    gamma: float = 1.5  # Death rate
    delta: float = 0.75  # Conversion efficiency
    alive: bool = True
    born_at_step: int = field(default=None)
    died_at_step: int = field(default=None)
    thetas: list[float] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    vocalizations: list[str] = field(default_factory=list)

    def handle_birth(self, step: int):
        self.born_at_step = step
        self.alive = True
        self.thetas = []
        self.explanations = []
        self.vocalizations = []

    def handle_death(self, step: int):
        self.died_at_step = step
        self.alive = False

    def decide_theta(
        self,
        s: float,
        w: float,
        s_max: float,
        step: int,
        respond_verbosely: bool = True,
    ) -> float:
        if not self.alive:
            return self.thetas[-1] if self.thetas else 1.0

        wolf_resp = get_wolf_response(
            s=s,
            w=w,
            s_max=s_max,
            old_theta=self.thetas[-1] if self.thetas else 1.0,
            step=step,
            respond_verbosely=respond_verbosely,
        )

        self.thetas.append(wolf_resp.theta)
        self.explanations.append(wolf_resp.explanation)
        self.vocalizations.append(wolf_resp.vocalization)

        return wolf_resp.theta

    def process_step_wolf(self, params, state, step):
        """
        Process a single step for this wolf, calculating its contribution to the
        population dynamics.
        """
        if not self.alive:
            return state

        dt = params.get("dt", 0.02)
        s = state.get("s_state", 0)

        # Get the current theta (hunting intensity)
        current_theta = self.thetas[-1] if self.thetas else 1.0

        # Calculate wolf's contribution to population change
        # Wolf death rate
        dw_dt = -1 * self.gamma * 1  # gamma * this wolf

        # Wolf reproduction based on predation
        dw_dt += (
            self.delta  # conversion efficiency
            * self.beta  # predation rate
            * current_theta  # hunting intensity
            * s  # sheep population
            * 1  # this wolf
        )

        # Scale by dt and add to accumulator
        added_dw = dw_dt * dt
        state["step_accumulated_dw"] = state.get("step_accumulated_dw", 0) + added_dw

        # Calculate effect on sheep population (predation)
        ds_dt_one_w_only = -self.beta * current_theta * s
        state["step_accumulated_ds"] = state.get("step_accumulated_ds", 0) + (
            ds_dt_one_w_only * dt
        )

        return state


class Agents:
    def __init__(
        self,
        n_wolves: int = 10,
        beta: float = 0.1,
        gamma: float = 1.5,
        delta: float = 0.75,
    ):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.wolves: list[Wolf] = []
        for i in range(n_wolves):
            self.wolves.append(Wolf(wolf_id=i, beta=beta, gamma=gamma, delta=delta))

    def get_all_thetas(self) -> list[list[float]]:
        return [wolf.thetas for wolf in self.wolves if wolf.alive]

    def get_step_thetas(self, step: int) -> list[float]:
        # Added check to ensure step is within range for each wolf
        return [
            wolf.thetas[step]
            for wolf in self.wolves
            if wolf.alive and step < len(wolf.thetas)
        ]

    def birth_wolves(self, step: int, n_wolves: int) -> None:
        # Use the parameters stored at the Agents level
        for _ in range(n_wolves):
            new_wolf = Wolf(
                wolf_id=len(self.wolves),
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
            )
            new_wolf.handle_birth(step)
            self.wolves.append(new_wolf)

    def kill_wolves(self, step: int, n_wolves: int) -> None:
        living_wolves = [wolf for wolf in self.wolves if wolf.alive]
        wolves_to_kill = min(n_wolves, len(living_wolves))

        for i in range(wolves_to_kill):
            living_wolves[-(i + 1)].handle_death(step)

    def process_step(self, params, state, step) -> dict:
        # Reset accumulators for this step
        state["step_accumulated_dw"] = 0
        state["step_accumulated_ds"] = 0

        # Get current state values needed for decisions
        s = state.get("s_state", 0)
        s_max = params.get("s_max", 100)
        living_wolves_count = sum(1 for wolf in self.wolves if wolf.alive)

        # Process each wolf (decide theta and update state)
        # Shuffle the wolves to avoid any bias in the order of processing
        random.shuffle(self.wolves)
        for wolf in self.wolves:
            if wolf.alive:
                # First decide theta based on current state
                wolf.decide_theta(s, living_wolves_count, s_max, step)

                # Then update state based on new theta
                state = wolf.process_step(params, state, step)

        return state
