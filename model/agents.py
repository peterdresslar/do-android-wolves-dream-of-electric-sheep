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
    starting_theta: float = 0.5  # Default starting theta
    # All thetas for every step (always populated)
    thetas: list[float] = field(default_factory=list)
    # Decision history (only populated when a decision is made)
    decision_history: dict = field(
        default_factory=lambda: {
            "history_steps": [],  # step numbers when decisions happened
            "new_thetas": [],  # theta values chosen at those steps
            "explanations": [],  # explanations for those decisions
            "vocalizations": [],  # vocalizations for those decisions
            "prompts": [],  # prompts used for those decisions
        }
    )

    def to_dict(self) -> dict:
        """
        Convert wolf to a dictionary for serialization.
        """
        return {
            "wolf_id": self.wolf_id,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "alive": self.alive,
            "born_at_step": self.born_at_step,
            "died_at_step": self.died_at_step,
            "thetas": self.thetas,
            "history_steps": self.decision_history["history_steps"],
            "new_thetas": self.decision_history["new_thetas"],
            "prompts": self.decision_history["prompts"],
            "explanations": self.decision_history["explanations"],
            "vocalizations": self.decision_history["vocalizations"],
        }

    def alive_at_step(self, step: int) -> bool:
        """
        Check if the wolf was alive at a given step.
        """
        # Born before or on step AND (still alive OR died after this step)
        return (
            self.born_at_step is not None
            and self.born_at_step <= step
            and (self.died_at_step is None or self.died_at_step > step)
        )
    
    def handle_starting_theta(self, theta: float | None):
        """
        Handle the starting theta for a wolf. This is important
        as wolves could be born (depending on where we go from here)
        at various intertemporal points. Additionally, we may want to deal with 
        randomized starting thetas (and in fact we will to start with).
        """
        if theta is not None:
            self.starting_theta = theta
        else:
            self.starting_theta = round(random.uniform(0, 1), 2)

    def handle_birth(self, step: int, theta: float | None):
        self.born_at_step = step
        self.alive = True
        self.thetas = []
        self.decision_history = {
            "history_steps": [],
            "new_thetas": [],
            "explanations": [],
            "vocalizations": [],
            "prompts": [],
        }
        
        # Set the starting theta
        self.handle_starting_theta(theta)
        
 

    def handle_death(self, step: int):
        self.alive = False
        self.died_at_step = step

    def copy_theta(self) -> None:
        """Copy the most recent theta value to the next step"""
        if self.thetas:
            self.thetas.append(self.thetas[-1])
        else:
            # If no previous theta exists, use the starting theta
            self.thetas.append(self.starting_theta)

    def set_theta(self, step: int, theta: float):  # no ai
        """Set a fixed theta value for this wolf (used in no_ai mode)"""
        if theta is not None:
            self.thetas.append(theta)
            self.decision_history["history_steps"].append(step)
            self.decision_history["new_thetas"].append(theta)
            self.decision_history["prompts"].append("N/A")  # No prompt generated
            self.decision_history["explanations"].append(
                f"Using provided theta: {theta}"
            )
            self.decision_history["vocalizations"].append(
                f"Using provided theta: {theta}"
            )
            return theta
        else:
            # If no theta is provided, use the previous value or the starting theta
            self.copy_theta()
            return self.thetas[-1]

    def decide_theta(
        self,
        s: float,
        w: float,
        sheep_max: float,
        step: int,
        respond_verbosely: bool = True,
    ) -> float:
        """
        Decide the theta for this wolf.
        If AI is enabled, the theta is decided by the wolf using an LLM response.
        If AI is disabled, the theta is provided as an argument.

        Args:
            s: current sheep population
            w: current wolf population
            sheep_max: maximum sheep capacity
            step: current simulation step
            respond_verbosely: include verbose response from LLM
            theta: fixed theta value if provided

        Returns:
            The chosen theta value.
        """
        if not self.alive:
            return self.thetas[-1] if self.thetas else self.starting_theta

        # Call LLM to decide theta
        wolf_resp = get_wolf_response(
            s=s,
            w=w,
            sheep_max=sheep_max,
            old_theta=self.thetas[-1] if self.thetas else self.starting_theta,
            step=step,
            respond_verbosely=respond_verbosely,
        )

        self.thetas.append(wolf_resp.theta)
        self.decision_history["history_steps"].append(step)
        self.decision_history["new_thetas"].append(wolf_resp.theta)
        self.decision_history["prompts"].append(wolf_resp.prompt)
        self.decision_history["explanations"].append(wolf_resp.explanation)
        self.decision_history["vocalizations"].append(wolf_resp.vocalization)

        return wolf_resp.theta

    def process_step(self, params, domain, step):
        """
        Process a single step for this wolf, calculating its contribution to the
        population dynamics.

        Returns:
            A dictionary with the wolf's contributions to population changes
        """
        if not self.alive:
            return {"dw": 0, "ds": 0}

        dt = params.get("dt", 0.02)
        s = domain.sheep_state  # Get sheep state from domain

        # Get the current theta (hunting intensity)
        current_theta = self.thetas[-1] if self.thetas else self.starting_theta

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

        # Scale by dt
        added_dw = dw_dt * dt

        # Calculate effect on sheep population (predation)
        ds_dt_one_w_only = -self.beta * current_theta * s
        added_ds = ds_dt_one_w_only * dt

        return {"dw": added_dw, "ds": added_ds}


class Agents:
    def __init__(
        self,
        beta: float = 0.1,
        gamma: float = 1.5,
        delta: float = 0.75,
        opts: dict[str, Any] = field(default_factory=dict),
    ):
        """
        Minimal initialization for Agents class.
        Most initialization logic is handled by the create_agents factory method
        since this class will be instantiated once per model run.
        """
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.wolves: list[Wolf] = []
        self.opts = opts
        self.churn_rate = opts.get("churn_rate", 0.05)
        self.average_thetas: list[float] = []

    @staticmethod
    def create_agents(
        n_wolves: int = 10,
        beta: float = 0.1,
        gamma: float = 1.5,
        delta: float = 0.75,
        theta: float = None,
        opts: dict[str, Any] = None,
        initial_step: int = 0,
    ) -> "Agents":
        """
        Create a new Agents instance with n_wolves.
        """
        if opts is None:
            opts = {}
        
        agents = Agents(beta=beta, gamma=gamma, delta=delta, opts=opts)
        
        # Initialize with proper theta values
        default_theta = theta if theta is not None else 0.5  # Use 0.5 as default
        
        for i in range(n_wolves):
            wolf = Wolf(
                wolf_id=i,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )
            wolf.handle_birth(initial_step, default_theta)
            agents.wolves.append(wolf)
        
        # Initialize average theta history with the initial theta
        agents.average_thetas = [default_theta]
        
        return agents

    @property
    def wolves_count(self) -> int:
        return len(self.wolves)

    @property
    def living_wolves_count(self) -> int:
        return sum(1 for wolf in self.wolves if wolf.alive)

    def get_living_wolves(self) -> list[Wolf]:
        return [wolf for wolf in self.wolves if wolf.alive]

    def get_all_thetas(self) -> list[list[float]]:
        return [wolf.thetas for wolf in self.wolves if wolf.alive]

    def get_step_thetas(self, step: int) -> list[float]:
        # Added check to ensure step is within range for each wolf
        return [wolf.thetas[step] for wolf in self.wolves if wolf.alive_at_step(step)]

    def get_current_thetas(self) -> list[float]:
        """
        Get the current thetas of all living wolves.
        """
        return [wolf.thetas[-1] for wolf in self.wolves if wolf.alive]

    def get_mean_theta(self) -> float:
        """
        Get the mean current theta of all living wolves.
        """
        if not len(self.get_current_thetas()) == 0: # if no wolves are alive, return 0.0
            return sum(self.get_current_thetas()) / len(self.get_current_thetas())
        else:
            return 0.0

    # History methods for convenience, working with data from
    # our attached collection of wolves.

    def get_wolves_count_step(self, step: int) -> int:
        """
        Get the number of wolves at a given step.
        """
        # born before or on step
        return sum(1 for wolf in self.wolves if wolf.born_at_step <= step)

    def get_living_wolves_count_step(self, step: int) -> int:
        """
        Get the number of living wolves at a given step.
        """
        # Use the alive_at_step method to check if each wolf was alive at this step
        return sum(1 for wolf in self.wolves if wolf.alive_at_step(step))

    def get_living_wolf_count_history(self) -> list[int]:
        """
        Get the history of living wolf counts for all steps in the simulation.
        """
        # Determine the maximum step from theta history length
        # This ensures we cover all simulation steps, not just birth/death events
        max_step = max(len(wolf.thetas) for wolf in self.wolves) - 1 if self.wolves else 0
        
        # Ensure we have at least as many steps as average_thetas
        if hasattr(self, 'average_thetas') and self.average_thetas:
            max_step = max(max_step, len(self.average_thetas) - 1)
        
        return [self.get_living_wolves_count_step(step) for step in range(max_step + 1)]

    def get_living_wolves_step(self, step: int) -> list[Wolf]:
        """
        Get the living wolves at a given step.
        """
        # for alive and born before or on step, return wolf_id
        return [
            wolf for wolf in self.wolves if wolf.alive and wolf.born_at_step <= step
        ]

    def get_living_wolves_history(self) -> list[list[Wolf]]:
        """
        Get the history of living wolves.
        This is a list of lists of wolves, one list per step.
        """
        return [self.get_living_wolves_step(step) for step in range(len(self.wolves))]

    def get_average_theta_history(self) -> list[float]:
        """
        Get the history of average theta values.

        Returns:
            A list of average theta values, one per step
        """
        return self.average_thetas

    def get_decision_makers_at_step(self, step: int) -> list[int]:
        """
        Get IDs of wolves that made decisions at a specific step.
        """
        decision_makers = []
        for wolf in self.wolves:
            if step in wolf.decision_history["history_steps"]:
                decision_makers.append(wolf.wolf_id)
        return decision_makers

    def get_decision_makers_history(self) -> list[list[int]]:
        """
        Get the history of decision makers.
        """
        return [
            self.get_decision_makers_at_step(step) for step in range(len(self.wolves))
        ]

    def get_wolf_history_data(self) -> dict:
        """
        Get the history of wolf data.
        """
        return {
            "wolves_count": self.get_wolves_count_history(),
            "living_wolves_count": self.get_living_wolves_count_history(),
            "living_wolves_history": self.get_living_wolves_history(),
        }

    def get_current_state(self) -> dict:
        """
        Get the current state of all wolves for snapshot creation.

        Returns:
            A dictionary containing:
            - wolves_count: Number of living wolves
            - current_thetas: List of current theta values for living wolves
            - mean_theta: Average theta value across all living wolves
        """
        return {
            "wolves_count": self.get_living_wolves_count(),
            "current_thetas": self.get_current_thetas(),
            "mean_theta": self.get_mean_theta(),
        }

    def get_agents_summary(self) -> list[dict]:
        """
        Get a summary of all wolves.

        Returns:
            A list of dictionaries, one per wolf.
        """
        return [wolf.to_dict() for wolf in self.wolves]

    # Note that heterogenous wolves would need additional logic
    def birth_wolves(self, step: int, n_wolves: int, theta: float = None) -> None:
        # Use the parameters stored at the Agents level
        for _ in range(n_wolves):
            new_wolf = Wolf(
                wolf_id=len(self.wolves),
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
            )
            new_wolf.handle_birth(step, theta)
            self.wolves.append(new_wolf)

    def kill_wolves(self, step: int, n_wolves: int) -> None:
        """
        For now we pick the oldest wolves, but this would not be a good strategy
        for heterogenous wolves.
        """
        num_wolves = min(n_wolves, self.living_wolves_count)

        # Sort wolves by ID (oldest first)
        wolves_by_oldest = sorted(
            [wolf for wolf in self.wolves if wolf.alive], key=lambda x: x.wolf_id
        )

        # Kill the oldest wolves first
        for wolf in wolves_by_oldest[:num_wolves]:
            wolf.handle_death(step)

    def update_average_theta(self, append=True) -> float:
        """
        Calculate the average theta of all living wolves.

        Args:
            append: Whether to append the result to average_thetas (True) or
                   just return it without modifying the history (False)

        Returns:
            The calculated average theta
        """

        if self.living_wolves_count == 0:
            # Return the last average or a default if no history
            new_avg = 0.0
        else:
            new_avg = (
                sum(
                    wolf.thetas[-1]
                    for wolf in self.get_living_wolves()  # most recent thetas in wolf histories
                )
                / self.living_wolves_count
            )

        if append:
            self.average_thetas.append(new_avg)

        return new_avg

    def handle_population_changes(self, net_wolves_change: int, step: int) -> None:
        """
        IMPORTANT: This function is called by the model, as it is trading data back and forth with
        the domain. This specifically means that wolf population changes happen
        *after* the individual wolf steps, which instead focus on agency.

        Handle wolf population changes based on the net change value.

        Args:
            net_wolves_change: Integer representing the net change in wolf population
            step: Current simulation step
        """
        if net_wolves_change > 0:
            self.birth_wolves(step, net_wolves_change)
        elif net_wolves_change < 0:
            self.kill_wolves(step, abs(net_wolves_change))

    def process_step_sync(self, params, domain, step) -> None:
        """
        Process the step for all wolves, updating the domain directly.
        With synchronous churn: only a percentage of wolves update their theta each step.
        """
        # Reset accumulators in the domain
        domain.reset_accumulators()

        # Get current state values needed for decisions
        sheep_state = domain.sheep_state
        sheep_max = domain.sheep_capacity
        living_wolves = self.get_living_wolves()

        # Determine which wolves will update their theta this step
        # Randomly select wolves based on churn rate

        if self.opts.get("no_ai", True):
            for wolf in living_wolves:
                wolf.set_theta(step, params.get("theta", 0.5))
        else:
            churn_count = max(1, int(self.living_wolves_count * self.churn_rate))
            wolves_to_update = random.sample(
                living_wolves, min(churn_count, self.living_wolves_count)
            )

            for wolf in living_wolves:
                if wolf in wolves_to_update:
                    # First decide theta based on current state
                    wolf.decide_theta(
                        sheep_state, self.living_wolves_count, sheep_max, step, True
                    )
                else:
                    wolf.copy_theta()

                # Then update domain based on new theta
                domain_changes = wolf.process_step(params, domain, step)
                domain.step_accumulated_dw += domain_changes["dw"]
                domain.step_accumulated_ds += domain_changes["ds"]

        # Calculate and store the average theta after all wolves have decided
        if self.average_thetas:
            print(
                f"Updating average theta for step {step}. Was: {self.average_thetas[-1]}"
            )
        self.update_average_theta(append=True)
