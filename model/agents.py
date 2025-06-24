# agents.py
import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

# Call LLM to decide theta asynchronously
from .utils.llm_utils import get_wolf_response_async

THREADS_DEFAULT = 10


@dataclass
class Wolf:
    wolf_id: int
    beta: float
    gamma: float
    delta: float
    alive: bool
    starting_theta: float
    born_at_step: int = field(default=None)
    died_at_step: int = field(default=None)
    # All thetas for every step (always populated regardless of churn rate)
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
    last_sheep_state: float = field(default=None)
    last_wolves_count: int = field(default=None)

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

    def handle_birth(self, step: int, theta: float):
        self.born_at_step = step  # Set the born_at_step attribute
        self.thetas = []
        self.decision_history = {
            "history_steps": [],
            "new_thetas": [],
            "explanations": [],
            "vocalizations": [],
            "prompts": [],
        }

        self.thetas.append(theta)

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

    def set_theta(self, step: int, domain, params: dict):  # no ai
        """
        Set theta value for this wolf (used in no_ai mode).

        Logic:
        1. If decision_mode is constant, use the theta_start provided in params
        2. If decision_mode is adaptive, calculate theta using the function that responds to prey scarcity

        Args:
            step: Current simulation step
            domain: Domain object containing sheep state and capacity
            params: Parameters dictionary for theta function configuration

        Returns:
            The set theta value
        """
        decision_mode = params.get("decision_mode")

        # Check if we're using constant theta
        if decision_mode == "constant":
            # Use theta_start as a constant value
            constant_theta = params["theta_start"]
            self.thetas.append(constant_theta)
            self.decision_history["history_steps"].append(step)
            self.decision_history["new_thetas"].append(constant_theta)
            self.decision_history["prompts"].append("N/A")  # No prompt generated
            self.decision_history["explanations"].append(
                f"Using constant theta_start: {constant_theta}"
            )
            self.decision_history["vocalizations"].append(
                f"Using constant theta_start: {constant_theta}"
            )
            return constant_theta

        elif decision_mode == "adaptive":
            k = params.get("k")
            if k is None:
                raise ValueError(
                    "Parameter 'k' is required for adaptive mode but was not provided"
                )

            sheep_max = domain.sheep_capacity
            epsilon = params.get("eps")
            sheep_state = domain.sheep_state

            calculated_theta = 1.0 / (1.0 + k * sheep_max / (sheep_state + epsilon))

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
        else:
            # This shouldn't happen if decision_mode validation is done properly
            raise ValueError(f"Invalid decision_mode: {decision_mode}")

    async def decide_theta_async(
        self,
        s: float,
        w: float,
        sheep_max: float,
        step: int,
        respond_verbosely: bool,
        delta_s: float,
        delta_w: float,
        prompt_type: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> float:
        """
        Async version of decide_theta.
        Decide the theta for this wolf using an async LLM call.
        """
        if not self.alive:
            return self.thetas[-1] if self.thetas else self.starting_theta

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
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.thetas.append(wolf_resp.theta)
        self.decision_history["history_steps"].append(step)
        self.decision_history["new_thetas"].append(wolf_resp.theta)
        self.decision_history["prompts"].append(wolf_resp.prompt)
        self.decision_history["explanations"].append(wolf_resp.explanation)
        self.decision_history["vocalizations"].append(wolf_resp.vocalization)

        return wolf_resp.theta

    def process_step(self, step, domain, params):
        """
        Process a single step for this wolf, calculating its contribution to the
        population dynamics.

        Returns:
            A dictionary with the wolf's contributions to population changes
        """
        if not self.alive:
            return {
                "dw": 0,
                "ds": 0,
            }  # dead wolves don't contribute to population change

        dt = params.get("dt")
        s = domain.sheep_state  # Get sheep state from domain

        # Get the current theta (hunting intensity)
        current_theta = self.thetas[-1] if self.thetas else self.starting_theta

        # Calculate wolf's contribution to population change
        # Wolf death rate
        dw_dt = -1 * self.gamma * 1  # gamma * this wolf

        # Wolf reproduction based on predation
        dw_dt += (
            self.delta  # conversion efficiency
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
        params: dict[str, Any],
        initial_step: int,
        threads: int = THREADS_DEFAULT,
    ):
        """
        Minimal initialization for Agents class.
        Most initialization logic is handled by the create_agents factory method
        since this class will be instantiated once per model run.
        """
        self.params = params
        self.initial_step = initial_step
        self.current_step = initial_step
        self.threads = threads
        self.wolves = []
        self.wolf_count_history = []
        self.average_thetas = []
        self.last_wolf_death_step = None

    @staticmethod
    def create_agents(
        w_start: int,
        decision_mode: str,
        beta: float,
        gamma: float,
        delta: float,
        dt: float,
        theta_start: float,
        randomize_theta: bool,
        eps: float,
        k: float | None,
        model_name: str | None,
        temperature: float | None,
        prompt_type: str | None,
        churn_rate: float | None,
        initial_step: int,
        threads: int = THREADS_DEFAULT,
    ) -> "Agents":
        """
        Create a new Agents instance with w_start wolves.
        """

        params = {
            "w_start": w_start,
            "decision_mode": decision_mode,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "dt": dt,
            "theta_start": theta_start,
            "randomize_theta": randomize_theta,
            "eps": eps,
            "k": k,
            "model_name": model_name,
            "temperature": temperature,
            "prompt_type": prompt_type,
            "churn_rate": churn_rate,
            "threads": threads,
        }

        # print("Creating agents with params:", params)

        agents = Agents(params, initial_step)

        # Initialize wolves with either random or fixed theta
        for i in range(w_start):
            # Determine starting theta based on randomization setting
            if randomize_theta:
                # Generate random theta
                starting_theta_value = random.uniform(0, 1)
            else:
                # Use the configured theta_start
                starting_theta_value = theta_start

            wolf = Wolf(
                wolf_id=i,
                beta=beta,
                gamma=gamma,
                delta=delta,
                alive=True,
                starting_theta=starting_theta_value,
                last_sheep_state=None,
                last_wolves_count=None,
            )
            wolf.handle_birth(initial_step, starting_theta_value)
            agents.wolves.append(wolf)

        # Initialize history with the t=0 state
        initial_avg_theta = agents.get_mean_theta()
        agents.wolf_count_history = [w_start]
        agents.average_thetas = [initial_avg_theta]

        return agents

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

    def get_living_wolf_count_history(self) -> list[int]:
        """
        Get the history of living wolf counts for all steps in the simulation.
        """
        return self.wolf_count_history

    def get_average_theta_history(self) -> list[float]:
        """
        Get the history of average theta values.

        Returns:
            A list of average theta values, one per step
        """
        return self.average_thetas

    def get_agents_summary(self) -> list[dict]:
        """
        Get a summary of all wolves.

        Returns:
            A list of dictionaries, one per wolf.
        """
        return [wolf.to_dict() for wolf in self.wolves]

    # Note that heterogenous wolves would need additional logic
    def birth_wolves(self, step: int, net_wolves_change: int) -> None:
        # Use the parameters stored at the Agents level
        for _ in range(net_wolves_change):
            # Determine starting theta based on randomization setting
            if self.params.get("randomize_theta", False):
                # Generate random theta
                starting_theta = random.uniform(0, 1)
            else:
                # Use the configured theta_start
                starting_theta = self.params.get("theta_start")

            # Create a complete Wolf object with all required parameters
            new_wolf = Wolf(
                wolf_id=len(self.wolves),
                beta=self.params.get("beta"),
                gamma=self.params.get("gamma"),
                delta=self.params.get("delta"),
                alive=True,  # Wolf is born alive
                starting_theta=starting_theta,  # Set the starting theta
                last_sheep_state=None,  # Initialize with None
                last_wolves_count=None,  # Initialize with None
                born_at_step=step,
                died_at_step=None,
            )

            # Now handle_birth can properly initialize the wolf's state
            new_wolf.handle_birth(step, starting_theta)
            self.wolves.append(new_wolf)

    def kill_wolves(self, step: int, net_wolves_change: int) -> None:
        """
        For now we pick the oldest wolves, but this would not be a good strategy
        for heterogenous wolves.
        """
        num_wolves = min(net_wolves_change, self.living_wolves_count)

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

        # Record the new wolf count after births/deaths
        self.wolf_count_history.append(self.living_wolves_count)

        # Check if wolves went extinct. Only fire the first time
        if self.living_wolves_count == 0 and self.last_wolf_death_step is None:
            print(f"Wolves went extinct at step {step}")
            self.last_wolf_death_step = step

    async def process_step_async(self, domain, step) -> None:
        """
        Process the step for all wolves, updating the domain directly.
        With churn: only a percentage of wolves update their theta each step,
        These updates are processed in parallel batches, and any wolf is not
        guaranteed to update in any specific order.
        """
        # Reset accumulators in the domain
        domain.reset_accumulators()

        decision_mode = self.params.get("decision_mode")

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

        if decision_mode == "constant" or decision_mode == "adaptive":
            for wolf in living_wolves:
                # Don't pass theta parameter at all to force using the function
                wolf.set_theta(step, domain, self.params)
                domain_changes = wolf.process_step(step, domain, self.params)
                domain.step_accumulated_dw += domain_changes["dw"]
                domain.step_accumulated_ds += domain_changes["ds"]
        else:
            # Simple adaptive churn with a minimum number of wolves updating
            w_start = self.params.get("w_start")  # Get initial wolf count from params
            min_wolves_to_update = max(
                1, w_start // 2
            )  # At least half the initial wolves

            # Calculate churn count with a minimum floor
            churn_count = max(
                min_wolves_to_update,  # Minimum number of wolves to update
                int(
                    self.living_wolves_count * self.params.get("churn_rate")
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
            max_threads = self.params.get("threads")
            prompt_type = self.params.get("prompt_type")

            # Get the model_name from params or model_names
            model = self.params.get("model_name")
            temperature = self.params.get("temperature")
            max_tokens = self.params.get("max_tokens")
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
                            model,
                            temperature,
                            max_tokens,
                        )
                    )

                # Wait for all tasks in this batch to complete
                await asyncio.gather(*tasks)

            # After all decisions are made, process the step for each wolf
            for wolf in living_wolves:
                domain_changes = wolf.process_step(step, domain, self.params)
                domain.step_accumulated_dw += domain_changes["dw"]
                domain.step_accumulated_ds += domain_changes["ds"]

        # Calculate and store the average theta after all wolves have decided
        self.update_average_theta(append=True)

    def process_step_sync(self, domain, step) -> None:
        """
        Synchronous wrapper around process_step_async.
        Process the step for all wolves, updating the domain directly.
        Works in both regular Python environments and Jupyter notebooks.
        """
        # Update the current step
        self.current_step = step

        try:
            # Check if we're in an existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a notebook or other environment with a running loop
                # Use nest_asyncio to allow nested event loops
                import nest_asyncio

                nest_asyncio.apply()
                loop.run_until_complete(self.process_step_async(domain, step))
            else:
                # Normal case - no running event loop
                loop.run_until_complete(self.process_step_async(domain, step))
        except RuntimeError:
            # If we can't get a running event loop, create a new one
            # This is the safest approach for most environments
            asyncio.run(self.process_step_async(domain, step))
