# agents.py

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from .utils import get_wolf_response

THREADS_DEFAULT = 10


@dataclass
class Wolf:
    wolf_id: int
    beta: float = None
    gamma: float = None
    delta: float = None
    alive: bool = True
    born_at_step: int = field(default=None)
    died_at_step: int = field(default=None)
    starting_theta: float = None
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
    last_sheep_state: float = None  # Add this field
    last_wolves_count: int = None  # Add this field

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
            self.starting_theta = random.uniform(0, 1)

    def handle_birth(self, step: int, theta: float | None = None):
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
        if theta is not None:
            self.handle_starting_theta(theta)
        else:
            self.handle_starting_theta(None)

        self.thetas.append(self.starting_theta)

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

    def set_theta(
        self, step: int, theta: float, domain, params: dict
    ):  # no ai
        """
        Set theta value for this wolf (used in no_ai mode).
        
        Logic:
        1. If theta_star is provided in params, use it as a constant
        2. Otherwise, calculate theta using the function that responds to prey scarcity
        
        Args:
            step: Current simulation step
            theta: Fixed theta value if provided
            domain: Domain object containing sheep state and capacity
            params: Parameters dictionary for theta function configuration
            
        Returns:
            The set theta value
        """
        # Check if theta_star is provided in params (not None)
        if "theta_star" in params and params["theta_star"] is not None:
            # Use theta_star as a constant value
            constant_theta = params["theta_star"]
            self.thetas.append(constant_theta)
            self.decision_history["history_steps"].append(step)
            self.decision_history["new_thetas"].append(constant_theta)
            self.decision_history["prompts"].append("N/A")  # No prompt generated
            self.decision_history["explanations"].append(
                f"Using constant theta_star: {constant_theta}"
            )
            self.decision_history["vocalizations"].append(
                f"Using constant theta_star: {constant_theta}"
            )
            return constant_theta
        
        else:
            # Calculate theta using the function: θ(s) = 1/(1 + k*s0/(s + ε))
            k = params.get("k", 1.0)  # Sensitivity parameter
            s0 = params.get(
                "sheep_max", domain.sheep_capacity
            )  # Reference sheep population
            epsilon = params.get(
                "eps", 0.0001
                )  # Small constant to avoid division by zero
            sheep_state = domain.sheep_state

            # Calculate theta using the function from the methods notebook
            calculated_theta = 1.0 / (1.0 + k * s0 / (sheep_state + epsilon))

            print(f"  Calculated theta: {calculated_theta:.4f}")

            self.thetas.append(calculated_theta)
            self.decision_history["history_steps"].append(step)
            self.decision_history["new_thetas"].append(calculated_theta)
            self.decision_history["prompts"].append("N/A")  # No prompt generated
            self.decision_history["explanations"].append(
                f"Calculated theta: {calculated_theta:.4f} based on sheep population: {sheep_state:.2f}"
            )
            self.decision_history["vocalizations"].append(
                f"Calculated theta: {calculated_theta:.4f} based on sheep population: {sheep_state:.2f}"
            )
            return calculated_theta


    async def decide_theta_async(
        self,
        s: float,
        w: float,
        sheep_max: float,
        step: int,
        respond_verbosely: bool = True,
        delta_s: float = 0,
        delta_w: float = 0,
        prompt_type: str = "high",
        model: str = None,  # Add model parameter
    ) -> float:
        """
        Async version of decide_theta.
        Decide the theta for this wolf using an async LLM call.
        """
        if not self.alive:
            return self.thetas[-1] if self.thetas else self.starting_theta

        # Call LLM to decide theta asynchronously
        from .utils import get_wolf_response_async

        wolf_resp = await get_wolf_response_async(
            s=s,
            w=w,
            sheep_max=sheep_max,
            old_theta=self.thetas[-1] if self.thetas else self.starting_theta,
            step=step,
            respond_verbosely=respond_verbosely,
            delta_s=delta_s,
            delta_w=delta_w,
            prompt_type=prompt_type,
            model=model,  # Pass model parameter
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
        beta: float = None,
        gamma: float = None,
        delta: float = None,
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
        self.churn_rate = opts.get("churn_rate")
        self.average_thetas: list[float] = []

    @staticmethod
    def create_agents(
        n_wolves: int = None,
        beta: float = None,
        gamma: float = None,
        delta: float = None,
        theta: float = None,
        opts: dict[str, Any] = None,
        initial_step: int = None,
    ) -> "Agents":
        """
        Create a new Agents instance with n_wolves.
        """
        if opts is None:
            opts = {}

        agents = Agents(beta=beta, gamma=gamma, delta=delta, opts=opts)

        # Initialize with proper theta values
        default_theta = theta if theta is not None else 0.5  # Ensure default_theta is never None

        for i in range(n_wolves):
            wolf = Wolf(
                wolf_id=i,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )
            wolf.handle_birth(initial_step, default_theta)  # Pass default_theta directly
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
        return [wolf.thetas[-1] for wolf in self.get_living_wolves()]

    def get_mean_theta(self) -> float:
        """
        Get the mean current theta of all living wolves.
        """
        if (
            not len(self.get_current_thetas()) == 0
        ):  # if no wolves are alive, return 0.0
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
        max_step = (
            max(len(wolf.thetas) for wolf in self.wolves) - 1 if self.wolves else 0
        )

        # Ensure we have at least as many steps as average_thetas
        if hasattr(self, "average_thetas") and self.average_thetas:
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

    async def process_step_async(self, params, domain, step) -> None:
        """
        Process the step for all wolves, updating the domain directly.
        With asynchronous churn: only a percentage of wolves update their theta each step,
        but these updates are processed in parallel batches.
        """
        # Reset accumulators in the domain
        domain.reset_accumulators()

        # Get current state values needed for decisions
        sheep_state = domain.sheep_state
        sheep_max = domain.sheep_capacity
        living_wolves = self.get_living_wolves()
        living_wolves_count = self.living_wolves_count

        # Calculate deltas for sheep and wolves
        delta_s = 0
        delta_w = 0

        # If we have previous state, calculate deltas
        if hasattr(self, "last_sheep_state") and self.last_sheep_state is not None:
            delta_s = sheep_state - self.last_sheep_state

        if hasattr(self, "last_wolves_count") and self.last_wolves_count is not None:
            delta_w = living_wolves_count - self.last_wolves_count

        # Store current state for next step's delta calculation
        self.last_sheep_state = sheep_state
        self.last_wolves_count = living_wolves_count

        if self.opts.get("no_ai", True):
            # No AI mode - use set_theta with domain and params to calculate theta
            # Make sure we're passing the sheep_max parameter
            if "sheep_max" not in params:
                params = params.copy()  # Create a copy to avoid modifying the original
                params["sheep_max"] = sheep_max

   

            for wolf in living_wolves:
                # Don't pass theta parameter at all to force using the function
                wolf.set_theta(step, None, domain, params)
                domain_changes = wolf.process_step(params, domain, step)
                domain.step_accumulated_dw += domain_changes["dw"]
                domain.step_accumulated_ds += domain_changes["ds"]
        else:
            # Simple adaptive churn with a minimum number of wolves updating
            initial_wolf_count = params.get(
                "initial_wolves", 10
            )  # Get initial wolf count from params
            min_wolves_to_update = max(
                1, initial_wolf_count // 2
            )  # At least half the initial wolves

            # Calculate churn count with a minimum floor
            churn_count = max(
                min_wolves_to_update,  # Minimum number of wolves to update
                int(
                    self.living_wolves_count * self.churn_rate
                ),  # Standard churn calculation
            )

            # Ensure we don't try to update more wolves than exist
            churn_count = min(churn_count, self.living_wolves_count)

            wolves_to_update = random.sample(living_wolves, churn_count)

            # For wolves not updating, just copy their previous theta
            for wolf in living_wolves:
                if wolf not in wolves_to_update:
                    wolf.copy_theta()

            # Process wolves in batches to respect thread limit
            max_threads = params.get("threads", THREADS_DEFAULT)
            prompt_type = self.opts.get("prompt_type", "high")

            # Get the model_name from params or model_names
            model = None
            if (
                "model_names" in params
                and isinstance(params["model_names"], list)
                and params["model_names"]
            ):
                model = params["model_names"][0]

            # Process wolves in batches
            for i in range(0, len(wolves_to_update), max_threads):
                batch = wolves_to_update[i : i + max_threads]

                # Create tasks for each wolf in the batch
                tasks = []
                for wolf in batch:
                    tasks.append(
                        wolf.decide_theta_async(
                            sheep_state,
                            self.living_wolves_count,
                            sheep_max,
                            step,
                            True,
                            delta_s,
                            delta_w,
                            prompt_type,
                            model,  # Pass model parameter
                        )
                    )

                # Wait for all tasks in this batch to complete
                await asyncio.gather(*tasks)

            # After all decisions are made, process the step for each wolf
            for wolf in living_wolves:
                domain_changes = wolf.process_step(params, domain, step)
                domain.step_accumulated_dw += domain_changes["dw"]
                domain.step_accumulated_ds += domain_changes["ds"]

        # Calculate and store the average theta after all wolves have decided
        self.update_average_theta(append=True)

    def process_step_sync(self, params, domain, step) -> None:
        """
        Synchronous wrapper around process_step_async.
        Process the step for all wolves, updating the domain directly.
        Works in both regular Python environments and Jupyter notebooks.
        """
        try:
            # Check if we're in an existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a notebook or other environment with a running loop
                # Use nest_asyncio to allow nested event loops
                import nest_asyncio

                nest_asyncio.apply()
                loop.run_until_complete(self.process_step_async(params, domain, step))
            else:
                # Normal case - no running event loop
                loop.run_until_complete(self.process_step_async(params, domain, step))
        except RuntimeError:
            # If we can't get a running event loop, create a new one
            # This is the safest approach for most environments
            asyncio.run(self.process_step_async(params, domain, step))
