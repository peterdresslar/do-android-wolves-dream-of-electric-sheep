# agents.py

from dataclasses import dataclass, field
from typing import List
import random
from .utils import get_wolf_response

@dataclass
class Wolf:
    wolf_id: int
    theta: float = 1.0
    alive: bool = True

    # We can store these if we want to log or analyze them
    explanation: str = field(default=None)
    vocalization: str = field(default=None)

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
        # If the wolf is "dead," you might decide to skip or do something special
        if not self.alive:
            return self.theta

        # 1. Call the LLM using your `utils.get_wolf_response`
        wolf_resp = get_wolf_response(
            s=s,
            w=w,
            s_max=s_max,
            old_theta=self.theta,
            step=step,
            respond_verbosely=respond_verbosely
        )

        # 2. Update this wolf's internal state with the new decision
        self.theta = wolf_resp.theta
        self.explanation = wolf_resp.explanation
        self.vocalization = wolf_resp.vocalization

        # 3. (Optional) You could add some logic for "survival checks" or other updates
        # if self.theta < 0.01:
        #     self.alive = False  # Example condition for a wolf giving up, etc.

        return self.theta


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

    def get_thetas(self) -> List[float]:
        """
        Return a list of the current theta values for all wolves.
        """
        return [wolf.theta for wolf in self.wolves if wolf.alive]

    def remove_dead_wolves(self) -> None:
        """
        Utility to purge any wolves that aren't alive anymore.
        """
        self.wolves = [wolf for wolf in self.wolves if wolf.alive]
