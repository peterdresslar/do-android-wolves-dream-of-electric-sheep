# agents.py

import random
from dataclasses import dataclass, field
from typing import Any

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
    # All thetas for every step (always populated)
    thetas: list[float] = field(default_factory=list)
    # Decision history (only populated when a decision is made)
    decision_history: dict = field(default_factory=lambda: {
        'history_steps': [],       # step numbers when decisions happened
        'new_thetas': [],  # theta values chosen at those steps
        'explanations': [], # explanations for those decisions
        'vocalizations': [], # vocalizations for those decisions
        'prompts': []      # prompts used for those decisions
    })

    def handle_birth(self, step: int):
        self.born_at_step = step
        self.alive = True
        self.thetas = []
        self.decision_history = {
            'history_steps': [],
            'new_thetas': [],
            'explanations': [],
            'vocalizations': [],
            'prompts': []
        }

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
        theta: float | None = None,
        record_decision: bool = True,
    ) -> float:
        """
        Decide the theta for this wolf.
        If AI is enabled, the theta is decided by the wolf using an LLM response.
        If AI is disabled, the theta is provided as an argument.

        Args:
            s: current sheep population
            w: current wolf population
            s_max: maximum sheep capacity
            step: current simulation step
            respond_verbosely: include verbose response from LLM
            theta: fixed theta value if provided
            record_decision: whether to record this as a decision in history

        Returns:
            The chosen theta value.
        """
        if not self.alive:
            return self.thetas[-1] if self.thetas else 1.0

        if theta is not None:
            self.thetas.append(theta)
            if record_decision:
                self.decision_history['history_steps'].append(step)
                self.decision_history['new_thetas'].append(theta)
                self.decision_history['prompts'].append("N/A")  # No prompt generated
                self.decision_history['explanations'].append(f"Using provided theta: {theta}")
                self.decision_history['vocalizations'].append(f"Using provided theta: {theta}")
            return theta

        # Call LLM to decide theta
        wolf_resp = get_wolf_response(
            s=s,
            w=w,
            s_max=s_max,
            old_theta=self.thetas[-1] if self.thetas else 1.0,
            step=step,
            respond_verbosely=respond_verbosely,
        )

        self.thetas.append(wolf_resp.theta)
        if record_decision:
            self.decision_history['history_steps'].append(step)
            self.decision_history['new_thetas'].append(wolf_resp.theta)
            self.decision_history['prompts'].append(wolf_resp.prompt)
            self.decision_history['explanations'].append(wolf_resp.explanation)
            self.decision_history['vocalizations'].append(wolf_resp.vocalization)

        return wolf_resp.theta

    def process_step(self, params, domain, step):
        """
        Process a single step for this wolf, calculating its contribution to the
        population dynamics.
        """
        if not self.alive:
            return domain

        dt = params.get("dt", 0.02)
        s = domain.s_state

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
        domain.step_accumulated_dw += added_dw

        # Calculate effect on sheep population (predation)
        ds_dt_one_w_only = -self.beta * current_theta * s
        domain.step_accumulated_ds += ds_dt_one_w_only * dt

        return domain


class Agents:
    def __init__(
        self,
        n_wolves: int = 10,
        beta: float = 0.1,
        gamma: float = 1.5,
        delta: float = 0.75,
        theta: float = 0.5, # noqa: N806. Not used.
        opts: dict[str, Any] = field(default_factory=dict),
    ):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.wolves: list[Wolf] = []
        for i in range(n_wolves):
            self.wolves.append(Wolf(wolf_id=i, beta=beta, gamma=gamma, delta=delta))
        self.opts = opts

        # Add churn rate parameter with default of 5%
        # a la JM Applegate, 2018
        self.churn_rate = opts.get("churn_rate", 0.05)

        # Initialize with a starting value for step 0
        self.average_thetas: list[float] = []

        # Calculate initial average theta (will be 0 since wolves have no thetas yet)
        initial_avg = 0.0
        self.average_thetas.append(initial_avg)

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

    def calculate_average_theta(self) -> float:
        """Calculate the average theta of all living wolves."""
        living_wolves = [wolf for wolf in self.wolves if wolf.alive]
        if not living_wolves:
            # Return the last average or a default if no history
            return self.average_thetas[-1] if self.average_thetas else 0.0

        return sum(
            wolf.thetas[-1] if wolf.thetas else 0.0 for wolf in living_wolves
        ) / len(living_wolves)

    def process_step_sync(self, params, domain, step) -> None:
        """
        Process the step for all wolves, updating the domain directly.
        With synchronous churn: only a percentage of wolves update their theta each step.
        """
        # Reset accumulators in the domain
        domain.reset_accumulators()

        if self.opts.get("no_ai", False):
            theta = params.get("theta", 0.5)  # Use theta_star from params
        else:
            theta = None

        # Get current state values needed for decisions
        s = domain.s_state
        s_max = domain.sheep_capacity
        living_wolves = [wolf for wolf in self.wolves if wolf.alive]
        living_wolves_count = len(living_wolves)

        # Determine which wolves will update their theta this step
        # Randomly select wolves based on churn rate
        wolves_to_update = []
        if not self.opts.get("no_ai", False):  # Only apply churn if AI is enabled
            churn_count = max(1, int(living_wolves_count * self.churn_rate))
            wolves_to_update = random.sample(
                living_wolves, min(churn_count, living_wolves_count)
            )

        # Process each wolf (decide theta and update domain)
        # Shuffle the wolves to avoid any bias in the order of processing
        shuffled_wolves = list(self.wolves)  # Create a copy to shuffle
        random.shuffle(shuffled_wolves)

        for wolf in shuffled_wolves:
            if wolf.alive:
                # First decide theta based on current state
                if self.opts.get("no_ai", False):
                    # All wolves use the fixed theta in no_ai mode
                    wolf.decide_theta(s, living_wolves_count, s_max, step, False, theta, record_decision=True)
                elif wolf in wolves_to_update:
                    # Only selected wolves update their theta
                    wolf.decide_theta(s, living_wolves_count, s_max, step, True, record_decision=True)
                else:
                    # Other wolves keep their previous theta
                    if wolf.thetas:
                        previous_theta = wolf.thetas[-1]
                        # Add theta but don't record as a decision
                        wolf.decide_theta(s, living_wolves_count, s_max, step, False, previous_theta, record_decision=False)
                    else:
                        # If a wolf has no previous theta (e.g., newly born), give it a default
                        default_theta = params.get("theta_star", 0.5)
                        # Add theta but don't record as a decision
                        wolf.decide_theta(s, living_wolves_count, s_max, step, False, default_theta, record_decision=False)

                # Then update domain based on new theta
                wolf.process_step(params, domain, step)

        # Calculate and store the average theta after all wolves have decided
        avg_theta = self.calculate_average_theta()
        self.average_thetas.append(avg_theta)

    def get_agents_summary(self) -> list[dict]:
        """Return a summary of all wolf agents with key details."""
        return [
            {
                'wolf_id': wolf.wolf_id,
                'thetas': wolf.thetas,
                'alive': wolf.alive,
                'born_at_step': wolf.born_at_step,
                'died_at_step': wolf.died_at_step,
                'history_steps': wolf.decision_history['history_steps'],
                'new_thetas': wolf.decision_history['new_thetas'],
                'explanations': wolf.decision_history['explanations'],
                'vocalizations': wolf.decision_history['vocalizations'],
                'prompts': wolf.decision_history['prompts']
            } for wolf in self.wolves
        ]
