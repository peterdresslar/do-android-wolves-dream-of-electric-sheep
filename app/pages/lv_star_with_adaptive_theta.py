import streamlit as st
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
import pandas as pd
import altair as alt

from utils.lv_core import (
    lv_star_with_theta_ode_ivp,
    lv_star_with_adaptive_theta_ode_ivp,
    allee_sheep_event,
    allee_wolf_event,
    allee_terminal_event,
    reset_events,
)

# Constant DT: since our solver parameters are hardcoded it makes sense to also use a constant DT
DT = 0.02
# the following are solve_ivp tuning parameters
ATOL = 1e-8
RTOL = 1e-8
DENSE_OUTPUT = True

epsilon = 1e-8

# --- Analysis functions ---#


#--- Streamlit helper ---#
def reset_to_defaults() -> None:
    st.session_state.Time = 50.0
    st.session_state.s_start = 10
    st.session_state.w_start = 10
    st.session_state.alpha = 1.0
    st.session_state.beta = 1.0
    st.session_state.gamma = 1.0
    st.session_state.delta = 1.0
    st.session_state.K = 50
    st.session_state.A = 2
    st.session_state.k = 0.1
    st.session_state.disable_adaptive_theta = False

def init_example():
    reset_events()
    T = st.session_state.T
    alpha = st.session_state.alpha
    beta = st.session_state.beta
    gamma = st.session_state.gamma
    delta = st.session_state.delta
    s_start = st.session_state.s_start
    w_start = st.session_state.w_start
    A = st.session_state.A
    K = st.session_state.K
    k = st.session_state.k
    disable_adaptive_theta = st.session_state.disable_adaptive_theta

    t_eval = np.arange(0.0, T, DT)
    return T, t_eval, alpha, beta, gamma, delta, s_start, w_start, A, K, k, disable_adaptive_theta


# --- Streamlit page building ---#
def add_sidebar() -> None:
    st.sidebar.header("Controls")
    st.sidebar.button("Reset to defaults", on_click=reset_to_defaults)
    st.sidebar.slider(
        "Time", key="T", value=50.0, min_value=1.0, max_value=250.0, step=1.0
    )

    st.sidebar.slider("k", key="k", value=0.5, min_value=0.0, max_value=1.0, step=0.05)
    st.sidebar.checkbox("Disable adaptive theta (theta=1.0)", key="disable_adaptive_theta", value=False)


def add_example_1_sidebar() -> None:
    # group for example 1. the four params and two initial conditions
    # section title for sidebar
    st.sidebar.markdown("### Example 1")
    st.sidebar.number_input(
        "alpha", key="alpha", value=1.0, min_value=0.0, max_value=10.0, step=0.1
    )
    st.sidebar.number_input(
        "beta", key="beta", value=1.0, min_value=0.0, max_value=10.0, step=0.1
    )
    st.sidebar.number_input(
        "gamma", key="gamma", value=1.0, min_value=0.0, max_value=10.0, step=0.1
    )
    st.sidebar.number_input(
        "delta", key="delta", value=1.0, min_value=0.0, max_value=10.0, step=0.1
    )
    st.sidebar.number_input(
        "s_start", key="s_start", value=10, min_value=0, max_value=1000, step=1
    )
    st.sidebar.number_input(
        "w_start", key="w_start", value=10, min_value=0, max_value=1000, step=1
    )
    st.sidebar.number_input("K", key="K", value=100, min_value=0, max_value=500, step=1)
    st.sidebar.number_input("A", key="A", value=2, min_value=0, max_value=10, step=1)


def render_intro() -> None:
    st.markdown("## LV* with $\\theta$ as a control variable")


def render_analysis() -> None:
    st.markdown(
        r"""
    In the previous example, we saw that the system can be stabilized by setting $\theta$ to a constant value. However, this is not the only way to control the system. We can also make $\theta$ a function of the state of the system.
    This is the idea of adaptive control.

    In our implementation, we will make $\theta$ a function of the state of the system. This will be done by the following function:
    $$
    \theta = 1.0 / (1.0 + k * K / (s + epsilon))
    $$
    Where $k$ is a constant and $K$ is the carrying capacity of the system. (Unfortunately, $k$ and $K$ both idiomatic.)
    
    “We set \theta(s;k,K)=\frac{s}{s+kK}. The sensitivity k is the fraction of capacity at which predation is half ‘on’: \theta=\tfrac12 when s=kK. Smaller k ramps \theta sooner (more aggressive), larger k delays predation until prey is abundant (more conservative).”
    """
    )


def render_example_one() -> None:
    """
    Single chart with adaptive theta showing population dynamics and phase portrait
    """
    st.markdown("""
    ### Example 1: LV* with Adaptive θ(s; k, K)
    With adaptive control, θ varies with the sheep population, modulating predation intensity based on prey abundance.
    This allows the system to respond dynamically to changing conditions.
    """)
    
    T, t_eval, alpha, beta, gamma, delta, s_start, w_start, A, K, k, disable_adaptive_theta = init_example()

    with st.spinner("Running simulation..."):
        reset_events()
        # if disable_adaptive_theta is True, use the regular LV* with theta=1.0
        if disable_adaptive_theta:
            solution = solve_ivp(
                lv_star_with_theta_ode_ivp,
                [0.0, T],
                [s_start, w_start],
                args=(alpha, beta, gamma, delta, K, A, 1.0),
                method="RK45",
                t_eval=t_eval,
                rtol=RTOL,
                atol=ATOL,
                dense_output=DENSE_OUTPUT,
                events=[allee_sheep_event, allee_wolf_event, allee_terminal_event],
            )
        else:
            solution = solve_ivp(
                lv_star_with_adaptive_theta_ode_ivp,
                [0.0, T],
                [s_start, w_start],
                args=(alpha, beta, gamma, delta, K, A, k),
                method="RK45",
                t_eval=t_eval,
                rtol=RTOL,
                atol=ATOL,
                dense_output=DENSE_OUTPUT,
                events=[allee_sheep_event, allee_wolf_event, allee_terminal_event],
            )

        # Post-process to show entire time range even if simulation stopped early
        solution_df = pd.DataFrame(
            solution.y.T,
            index=solution.t,
            columns=["Sheep", "Wolves"],
        ).reindex(t_eval).fillna(0.0)
        
        # Compute theta values over time for visualization
        if disable_adaptive_theta:
            theta_values = [1.0 for s in solution.y[0]]
        else:
            theta_values = [(s + epsilon) / ((s + epsilon) + (k * K)) for s in solution.y[0]]
        theta_df = pd.DataFrame({
            "Time": np.arange(len(solution.t)),
            "θ": theta_values
        }).set_index("Time")

    # Time series plot
    st.caption("Figure 1: Population Dynamics with Adaptive Control")
    st.line_chart(solution_df, x_label="Time", y_label="Population Density")
    
    st.markdown(f"""
    The adaptive control function θ(s; k={k}, K={K}) modulates predation based on sheep abundance. 
    Below we show how θ varies over time in response to the sheep population.
    """)
    
    st.caption("Figure 2: Adaptive Control Signal θ(t)")
    st.line_chart(theta_df, x_label="Time", y_label="θ (Predation Control)")
    
    # Phase portrait
    st.markdown("""
    The phase portrait below shows the trajectory in (Sheep, Wolves) space. The Allee thresholds are shown as purple dashed lines.
    """)
    
    phase_df = pd.DataFrame({
        "t": solution.t,
        "Sheep": solution.y[0],
        "Wolves": solution.y[1],
    })
    
    phase_chart = (
        alt.Chart(phase_df)
        .mark_line()
        .encode(
            x=alt.X("Sheep:Q", title="Sheep", sort=None),
            y=alt.Y("Wolves:Q", title="Wolves"),
            order="t:Q",
            tooltip=["t:Q", "Sheep:Q", "Wolves:Q"],
        )
        .properties(width="container")
    )
    
    st.caption("Figure 3: Phase Portrait")
    # Add Allee threshold lines and carrying capacity
    K_line = alt.Chart(pd.DataFrame({'x': [K]})).mark_rule(color='orange').encode(x='x:Q')
    allee_s = alt.Chart(pd.DataFrame({'x': [A]})).mark_rule(color='purple', strokeDash=[2, 2]).encode(x='x:Q')
    allee_w = alt.Chart(pd.DataFrame({'y': [A]})).mark_rule(color='purple', strokeDash=[2, 2]).encode(y='y:Q')
    
    st.altair_chart(phase_chart + K_line + allee_s + allee_w, use_container_width=True)
    st.caption("Orange: carrying capacity K; purple dashed: Allee thresholds A.")
    
    st.markdown(r"""
    As we can see, the adaptive control allows the system to modulate predation intensity dynamically. 
    When prey is scarce, θ → 0 (low predation), and when prey is abundant, θ → 1 (full predation).
    The sensitivity parameter k controls how quickly this transition occurs.
    
    Try adjusting k in the sidebar: smaller k makes the controller more aggressive (responds earlier), 
    while larger k makes it more conservative (waits for higher prey abundance).
    """
    )


def render_footer() -> None:
    st.divider()
    st.markdown("### Quick Navigation")
    st.page_link("pages/home.py", label="Home", icon="🏠")


def main() -> None:
    add_sidebar()
    add_example_1_sidebar()
    render_intro()
    render_analysis()
    render_example_one()

    render_footer()


main()
