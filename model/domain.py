from dataclasses import dataclass


@dataclass
class Domain:
    """
    Domain class to manage the state of the simulation.
    """

    def __init__(self, sheep_capacity: int, starting_sheep: int, starting_wolves: int):
        # Configuration parameters
        self.sheep_capacity = sheep_capacity
        self.starting_sheep = starting_sheep
        self.starting_wolves = starting_wolves
        
        # State variables
        self.s_state = starting_sheep
        
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
            "s_state": self.s_state,
            "step_accumulated_ds": self.step_accumulated_ds,
            "step_accumulated_dw": self.step_accumulated_dw,
            "accumulated_dw_remainder": self.accumulated_dw_remainder,
            "current_step": self.current_step
        }
    
    def update_from_state_dict(self, state_dict):
        """Update domain state from a state dictionary."""
        self.s_state = state_dict.get("s_state", self.s_state)
        self.step_accumulated_ds = state_dict.get("step_accumulated_ds", self.step_accumulated_ds)
        self.step_accumulated_dw = state_dict.get("step_accumulated_dw", self.step_accumulated_dw)
        self.accumulated_dw_remainder = state_dict.get("accumulated_dw_remainder", self.accumulated_dw_remainder)
        self.current_step = state_dict.get("current_step", self.current_step)
    
    def process_sheep_growth(self, params):
        """Process sheep growth according to the model parameters."""
        alpha = params.get("alpha", 1.0)
        dt = params.get("dt", 0.02)
        
        # Calculate sheep growth
        ds_dt = alpha * self.s_state
        self.s_state += ds_dt * dt
        
        # Apply sheep capacity limit
        self.s_state = min(self.s_state, self.sheep_capacity)
        self.s_state = max(self.s_state, 0)  # Ensure non-negative
    
    def process_wolf_effects(self, params):
        """Process the accumulated effects of wolves on the system."""
        dt = params.get("dt", 0.02)
        
        # Apply accumulated sheep change from predation
        self.s_state += self.step_accumulated_ds
        
        # Ensure sheep population stays within bounds
        self.s_state = min(self.s_state, self.sheep_capacity)
        self.s_state = max(self.s_state, 0)
        
        # Calculate net wolf population change
        dw_total = self.step_accumulated_dw + self.accumulated_dw_remainder
        
        # Add small epsilon if configured (for "last wolf" scenarios)
        eps = params.get("eps", 0)
        if eps > 0:
            dw_total += eps * dt
        
        # Calculate integer and fractional parts of wolf change
        net_wolves_change = int(dw_total)
        self.accumulated_dw_remainder = dw_total - net_wolves_change
        
        return net_wolves_change
    
    def increment_step(self):
        """Increment the current step counter."""
        self.current_step += 1


