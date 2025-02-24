from dataclasses import dataclass


@dataclass
class Domain:
    """
    Will talk about this later.
    """

    def __init__(self, sheep_capacity: int, starting_sheep: int, starting_wolves: int):
        self.sheep_capacity = sheep_capacity
        self.starting_sheep = starting_sheep
        self.starting_wolves = starting_wolves
        self.step_accumulated_ds = None
        self.step_accumulated_dw = None
        self.accumulated_dw_remainder = None
        self.s_state = None


