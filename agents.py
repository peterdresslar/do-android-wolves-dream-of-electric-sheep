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
        """
        Handle the birth of the wolf.
        """
        self.born_at_step = step
        self.alive = True
        self.thetas = []
        self.explanations = []
        self.vocalizations = []

    def handle_death(self, step: int):
        """
        Handle the death of the wolf.
        """
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
        """
        Each wolf calls the LLM to decide its new theta.

        Parameters
        ----------
        s : float
            Current number of sheep in the environment.
        w : float
            Current number of wolves in the environment (including this one).
        s_max : float
            Maximum capacity (upper bound) for sheep.
        step : int
            The current time step.
        respond_verbosely : bool
            If True, we ask for an explanation and a vocalization in the JSON.

        Returns
        -------
        new_theta : float
            The wolf's new theta, clamped to [0,1].
        """
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

        Parameters
        ----------
        params : dict
            Model parameters
        state : dict
            Current state of the simulation
        step : int
            Current time step

        Returns
        -------
        state : dict
            Updated state after this wolf's contribution
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
    """
    A container for multiple agents, Wolf or otherwise.
    """

    def __init__(
        self,
        n_wolves: int = 10,
        beta: float = 0.1,
        gamma: float = 1.5,
        delta: float = 0.75,
    ):
        """
        Initialize the pack with n_wolves, each having default theta=1.0.

        Parameters
        ----------
        n_wolves : int
            Number of wolves to create
        beta : float
            Predation rate parameter
        gamma : float
            Death rate parameter
        delta : float
            Conversion efficiency parameter
        """
        self.wolves: list[Wolf] = []
        for i in range(n_wolves):
            self.wolves.append(Wolf(wolf_id=i, beta=beta, gamma=gamma, delta=delta))

    def decide_all_synchronous(
        self,
        s: float,
        w: float,
        s_max: float,
        step: int,
        respond_verbosely: bool = True,
    ) -> None:
        """
        Iterate over each wolf and have it decide its theta.

        Parameters
        ----------
        s : float
            Current number of sheep.
        w : float
            Current total number of wolves (size of self.wolves).
        s_max : float
            Maximum sheep capacity.
        step : int
            Current time step.
        respond_verbosely : bool
            If True, each wolf is prompted for explanation & vocalization.
        """
        # We assume w is the count of living wolves or total wolves
        # For simplicity, let's use the length of self.wolves:
        total_wolves = len(self.wolves)

        for wolf in self.wolves:
            # shuffle the wolves
            random.shuffle(self.wolves)
            if wolf.alive:
                # Each wolf sees 'w' as total wolf count or living wolf count
                wolf.decide_theta(s, total_wolves, s_max, step, respond_verbosely)

    def get_all_thetas(self) -> list[float]:
        """
        Return a list of the current theta values for all wolves.
        """
        return [wolf.thetas for wolf in self.wolves if wolf.alive]

    def get_step_thetas(self, step: int) -> list[float]:
        """
        Return a list of the current theta values for all wolves that are alive.
        """
        return [wolf.thetas[step] for wolf in self.wolves if wolf.alive]

    def birth_wolves(self, step: int, n_wolves: int) -> None:
        """
        Birth n_wolves at the current step.
        """
        # Get parameters from an existing wolf if available
        if self.wolves:
            beta = self.wolves[0].beta
            gamma = self.wolves[0].gamma
            delta = self.wolves[0].delta
        else:
            # Default parameters if no wolves exist
            beta, gamma, delta = 0.1, 1.5, 0.75

        for _ in range(n_wolves):
            new_wolf = Wolf(
                wolf_id=len(self.wolves), beta=beta, gamma=gamma, delta=delta
            )
            new_wolf.handle_birth(step)
            self.wolves.append(new_wolf)

    def kill_wolves(self, step: int, n_wolves: int) -> None:
        """
        Kill n_wolves at the current step.
        """
        for _ in range(n_wolves):
            self.wolves[-1].handle_death(step)

    def process_step_agents(self, params, state, step) -> dict:
        """
        Process the step for all wolves, accumulating their effects on the system.

        Parameters
        ----------
        params : dict
            Model parameters
        state : dict
            Current state of the simulation
        step : int
            Current time step

        Returns
        -------
        state : dict
            Updated state after all wolves' contributions
        """
        # Reset accumulators for this step
        state["step_accumulated_dw"] = 0
        state["step_accumulated_ds"] = 0

        # Process each wolf's contribution
        for wolf in self.wolves:
            if wolf.alive:
                state = wolf.process_step_wolf(params, state, step)

        return state
