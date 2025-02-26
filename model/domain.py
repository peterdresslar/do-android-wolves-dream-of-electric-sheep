from dataclasses import dataclass


@dataclass
class Domain:
    """
    Domain class to manage the state of the simulation.
    """

    def __init__(self, sheep_capacity: int, starting_sheep: int):
        # Configuration parameters
        self.sheep_capacity = sheep_capacity
        self.starting_sheep = starting_sheep

        # State variables
        self.sheep_state = starting_sheep

        # History tracking (missing)
        self.sheep_history = [starting_sheep]

        # Accumulators for changes in each step
        self.step_accumulated_ds = 0
        self.step_accumulated_dw = 0
        self.accumulated_dw_remainder = 0

        # Step counter
        self.current_step = 0

    def reset_accumulators(self):
        """Reset accumulators at the beginning of each step."""
        self.step_accumulated_ds = 0
        self.step_accumulated_dw = 0

    def get_state_dict(self):
        """Return the current state as a dictionary for agents to use."""
        return {
            "sheep_state": self.sheep_state,
            "step_accumulated_ds": self.step_accumulated_ds,
            "step_accumulated_dw": self.step_accumulated_dw,
            "accumulated_dw_remainder": self.accumulated_dw_remainder,
            "current_step": self.current_step,
        }

    def update_from_state_dict(self, state_dict):
        """Update domain state from a state dictionary."""
        self.sheep_state = state_dict.get("sheep_state", self.sheep_state)
        self.step_accumulated_ds = state_dict.get(
            "step_accumulated_ds", self.step_accumulated_ds
        )
        self.step_accumulated_dw = state_dict.get(
            "step_accumulated_dw", self.step_accumulated_dw
        )
        self.accumulated_dw_remainder = state_dict.get(
            "accumulated_dw_remainder", self.accumulated_dw_remainder
        )
        self.current_step = state_dict.get("current_step", self.current_step)

    # In the reference notebook, this is ODE_accumulate_and_fit()
    # This function is here because our sheep our being operated on, and they belong as a population
    # to the domain.
    # The wolves are also fractionalized here, which could be designed out. But, it would more likely
    # be the case that we would simply agentize the sheep, and out this function would go.
    def accumulate_and_fit(self, params):
        ds_total = self.step_accumulated_ds
        dw_total = self.step_accumulated_dw
        accumulated_dw_remainder = self.accumulated_dw_remainder

        eps = params['eps']
        dt = params['dt']

        dw_total += accumulated_dw_remainder

        if eps and eps > 0:
            dw_total += eps * dt

        new_s = self.sheep_state + ds_total
        new_s_or_zero = max(0, new_s)
        new_s_or_max = min(new_s_or_zero, self.sheep_capacity)
        self.sheep_state = new_s_or_max
        self.sheep_history.append(self.sheep_state)

        net_wolves_change = int(dw_total)
        new_remainder = dw_total - net_wolves_change

        self.accumulated_dw_remainder = new_remainder

        return net_wolves_change

    # in the reference notebook, this is process_s_euler_forward()
    # This function is here because our sheep are not agentized and still behave as domain
    def process_sheep_growth(self, params):
        """Process sheep growth according to the model parameters."""
        alpha = params["alpha"]
        s = self.sheep_state

        ds_dt = alpha * s

        new_s = max(0, s + ds_dt * params["dt"])
        self.sheep_state = new_s

        # Apply sheep capacity limit
        self.sheep_state = min(self.sheep_state, self.sheep_capacity)
        self.sheep_history.append(self.sheep_state)

    def increment_step(self):
        """Increment the current step counter."""
        self.current_step += 1
