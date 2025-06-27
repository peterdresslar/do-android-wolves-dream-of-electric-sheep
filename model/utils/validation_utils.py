# validation_utils.py
"""
Demonstrate the partial and full discretization of the model,
the effect of the gather_and_fit approach, and compare to a standard ODE solver for the base LV system.

"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from model.model import run as run_model
from model.utils.simulation_utils import get_reference_ODE_with_cliff
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_lv_ode(model_params: dict, model_time: dict, cliff: str) -> pd.DataFrame:
    """
    Run the LV ODE model.
    """
    return get_reference_ODE_with_cliff(model_params, model_time, cliff_type=cliff)


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

    # Our histories now include the initial state plus N steps
    # So we have N+1 points total
    num_points = len(results["sheep_history"])

    # Generate time array to match: initial state at t=0, then N steps of size dt
    time_steps = [i * params["dt"] for i in range(num_points)]

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


def plot_forward_euler_vs_ode(t, s_euler, w_euler, s_exact, w_exact):
    # This is just two plots, with inputs from functions including the forward euler solver.
    plt.style.use('seaborn-v0_8-poster')
    plt.figure(figsize = (12, 8))

    plt.plot(t, s_euler, 'bo--', label='Approximate Sheep (Euler)')
    plt.plot(t, s_exact, 'g-', label='Exact Sheep (odeint)')
    plt.plot(t, w_euler, 'ro--', label='Approximate Wolves (Euler)')
    plt.plot(t, w_exact, 'm-', label='Exact Wolves (odeint)')

    plt.title('Forward Euler vs. odeint for Lotka-Volterra System')
    plt.xlabel('Time (t)')
    plt.ylabel('Population')
    plt.ylim(0, 25)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

    # 4. Create the Phase Space Plot
    plt.figure(figsize=(10, 10))
    plt.plot(s_euler, w_euler, 'bo--', label='Approximate (Euler)')
    plt.plot(s_exact, w_exact, 'g-', label='Exact (odeint)')
    plt.title('Phase Space: Wolves vs. Sheep')
    plt.xlabel('Sheep Population')
    plt.ylabel('Wolf Population')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


def plot_partial_matches_ode_with_cliff(model_params: dict, partial_results: pd.DataFrame, ode_results_cliff: pd.DataFrame, ode_results_no_cliff: pd.DataFrame):
     # --- Visualization ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle("ODE (cliff and no cliff) vs. Partial Discretization Comparison", fontsize=16)

    # Common y-limit for population plots
    y_limit = model_params["w_start"] * 10

    # Sheep population plot
    axes[0].plot(
        partial_results.index,
        partial_results["s"],
        label="Partial Discretization",
        color="blue",
    )
    axes[0].plot(
        ode_results_cliff.index,
        ode_results_cliff["s"],
        label="ODE (Cliff)",
        color="orange",
        linestyle="--",
    )
    axes[0].plot(
        ode_results_no_cliff.index,
        ode_results_no_cliff["s"],
        label="ODE (No Cliff)",
        color="green",
        linestyle=":",
    )
    axes[0].set_title("Sheep Population Over Time")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylim(0, y_limit)
    axes[0].set_ylabel("Population")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].xaxis.set_major_locator(mticker.MultipleLocator(100))

    # Wolf population plot
    axes[1].plot(
        partial_results.index,
        partial_results["w"],
        label="Partial Discretization",
        color="blue",
    )
    axes[1].plot(
        ode_results_cliff.index,
        ode_results_cliff["w"],
        label="ODE (Cliff)",
        color="orange",
        linestyle="--",
    )
    axes[1].plot(
        ode_results_no_cliff.index,
        ode_results_no_cliff["w"],
        label="ODE (No Cliff)",
        color="green",
        linestyle=":",
    )
    axes[1].set_title("Wolf Population Over Time")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylim(0, y_limit)
    axes[1].set_ylabel("Population")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].xaxis.set_major_locator(mticker.MultipleLocator(100))

    # --- Prepare data for Phase Space plot ---
    # Find the point of wolf extinction in the discrete model
    extinction_step = (partial_results['w'] == 0).idxmax()
    if extinction_step > 0:
        phase_space_results = partial_results.iloc[:extinction_step]
    else:
        # If wolves never go extinct, use the full dataset
        phase_space_results = partial_results

    # Phase space plot
    axes[2].plot(
        phase_space_results["s"],
        phase_space_results["w"],
        label="Partial Discretization",
        color="blue",
    )
    axes[2].plot(
        ode_results_cliff["s"],
        ode_results_cliff["w"],
        label="ODE (Cliff)",
        color="orange",
        linestyle="--",
    )
    axes[2].set_title("Phase Space")
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel("Sheep Population")
    axes[2].set_ylabel("Wolf Population")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def test_prop_partial_discretization_matches_ode(model_params: dict, lines: int = 5):
    """
    Hypothesis: the partial discretization should match the ODE model within error.

    Model parameters must be passed in as a dictionary. For instance:

    model_params = {
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 1.5,
        "delta": 0.75,
        "s_start": 100,
        "w_start": 10,
        "dt": 0.02,
        "steps": 1000,
        "sheep_max": 1000000,
        "eps": 0.0
    }

    We will run both the ode and partial discretization models for a given set of parameters and time.
    We will then compare the results.

    We will use the following metrics:
    - Mean absolute error
    - Root mean square error
    - R-squared

    """


    # The ODE solver needs N+1 points to match our discrete model
    # which includes the initial state plus N steps
    model_time = {
        "time": model_params["steps"] * model_params["dt"],  # Total time
        "tmax": model_params["steps"],  # Number of points (odeint will add t=0)
    }

    # Run both models
    ode_results_cliff = run_lv_ode(model_params, model_time, cliff="wolves")
    ode_results_no_cliff = run_lv_ode(model_params, model_time, cliff="none")
    partial_results = run_lv_partial_discretization(model_params)

    print("\n--- Model Results ---")
    # print(f"ODE results (Cliff) shape: {ode_results_cliff.shape}")
    # print(f"ODE results (No Cliff) shape: {ode_results_no_cliff.shape}")
    # print(f"Discrete results shape: {partial_results.shape}")

    print(f"\nODE Results (Cliff) (first {lines} rows):")
    print(ode_results_cliff.head(lines))
    print(f"\nPartial Discretization Results (first {lines} rows):")
    print(partial_results.head(lines))

    # Ensure all dataframes have the same length for comparison
    min_length = min(len(ode_results_cliff), len(ode_results_no_cliff), len(partial_results))
    ode_results_cliff = ode_results_cliff.iloc[:min_length]
    ode_results_no_cliff = ode_results_no_cliff.iloc[:min_length]
    partial_results = partial_results.iloc[:min_length]

    # --- Numerical Analysis vs. ODE with Cliff ---
    print("\n--- Numerical Comparison (Discrete vs. ODE with Cliff) ---")
    metrics_cliff = {}
    for pop in ["s", "w"]:
        y_true = ode_results_cliff[pop]
        y_pred = partial_results[pop]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred) if y_true.var() != 0 else 1.0
        metrics_cliff[pop] = {"mae": mae, "rmse": rmse, "r2": r2}

    for pop, m in metrics_cliff.items():
        pop_name = "Sheep" if pop == "s" else "Wolves"
        print(f"\n{pop_name} Population:")
        print(f"  Mean Absolute Error: {m['mae']:.4f}")
        print(f"  Root Mean Squared Error: {m['rmse']:.4f}")
        print(f"  R-squared: {m['r2']:.4f}")


    return ode_results_cliff, ode_results_no_cliff, partial_results


def compare_2(model_1: str, model_2: str, params: dict):
    """
    Compare two models.
    """
    pass
