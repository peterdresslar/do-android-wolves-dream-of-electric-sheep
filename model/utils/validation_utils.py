# validation_utils.py
"""
Demonstrate the partial and full discretization of the model,
the effect of the gather_and_fit approach, and compare to a standard ODE solver for the base LV system.

"""

import pandas as pd
from model.model import run as run_model
from model.utils.simulation_utils import get_reference_ODE


def run_lv_ode(model_params: dict, model_time: dict) -> pd.DataFrame:
    """
    Run the LV ODE model.
    """
    return get_reference_ODE(model_params, model_time)


def run_lv_partial_discretization(model_params: dict) -> pd.DataFrame:
    """
    Run the LV partial discretization (predator only).

    This runs the main simulation model with theta fixed at 1.0,
    which should approximate the behavior of the original ODEs.
    """
    params = model_params.copy()
    params["decision_mode"] = "constant"
    params["theta_start"] = 1.0
    params["randomize_theta"] = False  # Ensure theta is not randomized
    params["step_print"] = False
    params["save_results"] = False

    # The `run_model` function returns a dictionary of results.
    # We need to convert this to a DataFrame to match the ODE function's output.
    results = run_model(**params)

    # Ensure all histories have the same length
    max_len = max(
        len(results["sheep_history"]),
        len(results["wolf_history"]),
    )
    time_steps = [i * params["dt"] for i in range(max_len)]


    # Create a DataFrame
    df = pd.DataFrame(
        {
            "t": time_steps,
            "s": results["sheep_history"],
            "w": results["wolf_history"],
        }
    )
    return df


def run_lv_full_discretization(params: dict):
    """
    Run the LV full discretization (both predator and prey). Still in development.
    """
    pass


def test_prop_partial_discretization_matches_ode():
    """
    Hypothesis: the partial discretization should match the ODE model within error ??? with error sensitivity to time ???

    We will run both the ode and partial discretization models for a given set of parameters and time.
    We will then compare the results.

    We will use the following metrics:
    - Mean absolute error
    - Root mean square error
    - R-squared

    Finally, simple line plots as well as phase space plots will be provided.
    """
    # Define a base set of parameters for the comparison
    model_params = {
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 1.5,
        "delta": 0.75,
        "s_start": 100,
        "w_start": 10,
        "dt": 0.02,
        "steps": 1000,
        "sheep_max": 200,
        "eps": 0.0001,
    }

    model_time = {
        "time": int(model_params["steps"] * model_params["dt"]),
        "tmax": model_params["steps"],
    }

    # Run both models
    ode_results = run_lv_ode(model_params, model_time)
    partial_results = run_lv_partial_discretization(model_params)

    print("ODE Results:")
    print(ode_results.head())
    print("\nPartial Discretization Results:")
    print(partial_results.head())

    # NEXT: Add numerical and visual analysis here


def compare_2(model_1: str, model_2: str, params: dict):
    """
    Compare two models.
    """
    pass
