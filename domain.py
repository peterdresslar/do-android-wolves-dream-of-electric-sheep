from dataclasses import dataclass

@dataclass
class Domain:
    """
    The domain for this model is a set of "sheep" and the conditions in which they operate.
    Simply, sheep increase or decrease in number based upon:
    - The Lotka-Volterra functions for sheep growth, and
    - Wolf predation, which in our case is a function of our agents.
    The model does not have global discrete time, but the domain effectively has discrete time 
    for the convenience of operating sheep production.

    The domain is initialized by the model. The model then signals for the domain to begin processing
    with a start() method. The domain then processes for a number of discrete time steps, defined by
    the domain_step_size. The model and its agents can interact with the domain by eating sheep.
    """
    sheep_capacity: int
    current_sheep: int
    current_wolves: int
    domain_step_size: int # formalism: dt
    domain_time: int

    def __init__(self, sheep_capacity: int, current_sheep: int, domain_step_size: int):
        self.sheep_capacity = sheep_capacity
        self.current_sheep = current_sheep
        self.domain_step_size = domain_step_size

    def eat_sheep(self, mutton: int):
        self.current_sheep = max(0, self.current_sheep - mutton) # never negative sheep
    
    def process_discrete_time(self):
        pass

    def start():
        pass

    def stop():
        pass

    def reset():
        pass

    def get_state():
        pass

    def set_state():   
        pass

