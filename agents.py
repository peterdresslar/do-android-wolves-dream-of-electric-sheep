# agents.py

from dataclasses import dataclass, field
from typing import List
import random
from .utils import get_wolf_response

@dataclass
class Wolf:
    wolf_id: int
    alive: bool = True
    born_at_step: int = field(default=None)
    died_at_step: int = field(default=None)
    thetas: List[float] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    vocalizations: List[str] = field(default_factory=list)

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
        respond_verbosely: bool = True
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
            return self.theta

        wolf_resp = get_wolf_response(
            s=s,
            w=w,
            s_max=s_max,
            old_theta=self.thetas[-1],
            step=step,
            respond_verbosely=respond_verbosely
        )

        self.thetas.append(wolf_resp.theta)
        self.explanations.append(wolf_resp.explanation)
        self.vocalizations.append(wolf_resp.vocalization)

        return wolf_resp.theta


class Agents:
    """
    A container for multiple agents, Wolf or otherwise.
    """

    def __init__(self, n_wolves: int = 10):
        """
        Initialize the pack with n_wolves, each having default theta=1.0.
        """
        self.wolves: List[Wolf] = []
        for i in range(n_wolves):
            self.wolves.append(Wolf(wolf_id=i))

    def decide_all_synchronous(
        self,
        s: float,
        w: float,
        s_max: float,
        step: int,
        respond_verbosely: bool = True
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

    def get_all_thetas(self) -> List[float]:
        """
        Return a list of the current theta values for all wolves.
        """
        return [wolf.thetas for wolf in self.wolves if wolf.alive]
    
    def get_step_thetas(self, step: int) -> List[float]:
        """
        Return a list of the current theta values for all wolves that are alive.
        """
        return [wolf.thetas[step] for wolf in self.wolves if wolf.alive]

    def birth_wolves(self, step: int, n_wolves: int) -> None:
        """
        Birth n_wolves at the current step.
        """
        for i in range(n_wolves):
            self.wolves.append(Wolf(wolf_id=len(self.wolves)))
            self.wolves[-1].handle_birth(step)

    def kill_wolves(self, step: int, n_wolves: int) -> None:
        """
        Kill n_wolves at the current step.
        """
        for i in range(n_wolves):
            self.wolves[-1].handle_death(step)
