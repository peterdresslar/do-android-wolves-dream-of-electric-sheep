import streamlit as st
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
import pandas as pd

from utils.lv_core import lv_star_with_theta_ode_ivp, allee_sheep_event, allee_wolf_event, allee_terminal_event, reset_events

# Constant DT: since our solver parameters are hardcoded it makes sense to also use a constant DT
DT = 0.02
# the following are solve_ivp tuning parameters
ATOL = 1e-8
RTOL = 1e-8
DENSE_OUTPUT = True

# --- Analysis functions ---#


# --- Streamlit helper ---#
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

    t_eval = np.arange(0.0, T, DT)
    return T, t_eval, alpha, beta, gamma, delta, s_start, w_start, A, K


# --- Streamlit page building ---#
def add_sidebar() -> None:
    st.sidebar.header("Controls")
    st.sidebar.button("Reset to defaults", on_click=reset_to_defaults)
    st.sidebar.slider(
        "Time", key="T", value=50.0, min_value=1.0, max_value=250.0, step=1.0
    )


def add_example_1_sidebar() -> None:
    # group for example 1. the four params and two initial conditions
    # section title for sidebar
    st.sidebar.markdown("### Example 1")
    st.sidebar.number_input(
        "alpha", key="alpha", value=1.0, min_value=0.0, max_value=10.0, step=0.1
    )
    st.sidebar.number_input(
        "beta", key="beta", value=0.1, min_value=0.0, max_value=10.0, step=0.1
    )
    st.sidebar.number_input(
        "gamma", key="gamma", value=1.5, min_value=0.0, max_value=10.0, step=0.1
    )
    st.sidebar.number_input(
        "delta", key="delta", value=0.75, min_value=0.0, max_value=10.0, step=0.1
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
    We have observed that the stable oscillations of the standard Lotka-Volterra dynamical system are disrupted in a wide range of cases by the introduction of a simple but biologically plausible Allee threshold condition. 
    In real life population studies, however, the system successfully predicts population pair outcomes in a variety of ecologies, and with diverse parameter settings.
    Further, in Indvidual Based Modeling (IBM) techniques where the population is transformed from a single census variable into integerized unitary members, operations on a population below
    the specific threshold of two (or, taking things to a narrow extreme: one) seem particularly dischordant with respect to biological reality. Thus, we might be interested in ways
    to control the operations of the system in a manner that avoids the extinction threshold. 

    These motivations lead us to we explore the addition of a controlling parameter called $\theta$ to the LV* system. 
    
    In our implementation, we materialize an experimental system in which the predators are able (or compelled) to adjust their predation intensity in response to environmental factors. Relevant examples exist in the literature both 
    in theoretical models and in real-world observations. This control can be applied in the form of a single parameter, $\theta$, that modulates predation intensity $\beta$ in both the equations of the modified system.

    The system is given by the following differential equations:
    $$
    \begin{align}
    \frac{ds}{dt} &= \alpha s - \theta \beta s w \\
    \frac{dw}{dt} &= - \gamma w + \delta \theta \beta s w
    \end{align} \tag{3}
    $$

    Where $\theta \in [0, 1]$. 
    
    Observe that the equations in (3) imply that where $\theta$ is set to 0, predation is completely dampend; yielding a $\beta_{eff}$ of 0, with the resultant effects on both populations. When $\theta$ is set to 1, the operation of
    the system is identical to the LV* system with no control.

    Let's consider the "operation" of the control through different methods. Of course, the most simple of these methods would be to set $\theta$ to a constant value. This would be the equivalent of a "programmed" variable, in the sense that the value of $\theta$ is set before the simulation begins.
    """
    )


def render_example_one() -> None:
    """
    Four charts with constant theta values of 0, 0.2, 0.5, and 1.0.
    """
    T, t_eval, alpha, beta, gamma, delta, s_start, w_start, A, K = init_example()

    theta_values = [0.0, 0.2, 0.5, 1.0]
    
    with st.spinner("Running simulations and generating plots..."):
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # Flatten to make indexing easier
        
        for idx, theta in enumerate(theta_values):
            reset_events()  # Reset events before each of the four simulations (init_example also does this)
            solution = solve_ivp(
                lv_star_with_theta_ode_ivp,
                [0.0, T],
                [s_start, w_start],
                args=(
                    alpha,
                    beta,
                    gamma,
                    delta,
                    K,
                    A,
                    theta,
                ),
                method="RK45",
                t_eval=np.arange(0.0, T, DT),
                rtol=RTOL,
                atol=ATOL,
                dense_output=DENSE_OUTPUT,
                events=[allee_sheep_event, allee_wolf_event, allee_terminal_event] # events that change system trajectory
            )

            # Post-process: match Example 2's handling
            # Reindex to show entire time range even if simulation stopped early
            solution_df = pd.DataFrame(
                solution.y.T,
                index=solution.t,
                columns=["Sheep", "Wolves"],
            ).reindex(t_eval).fillna(0.0)

            # control x-axis: same time for all plots
            time_axis = np.arange(len(t_eval))
            
            # Get current axis
            ax = axes[idx]
            
            # Plot populations on left y-axis
            line1 = ax.plot(time_axis, solution_df["Sheep"], label="Sheep", color='blue', linewidth=2)
            line2 = ax.plot(time_axis, solution_df["Wolves"], label="Wolves", color='orange', linewidth=2)
            ax.set_xlabel("Time")
            ax.set_ylabel("Population Density")
            ax.set_title(f"θ = {theta}")
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line for Allee threshold A
            ax.axhline(y=A, color='purple', linestyle=':', linewidth=1, alpha=0.5, label=f'Allee threshold (A={A})')
            
            # Create right y-axis for theta
            ax2 = ax.twinx()
            # Plot constant theta line
            line3 = ax2.plot(time_axis, [theta] * len(time_axis), 
                            label=f'θ = {theta}', 
                            color='red', 
                            linestyle='--', 
                            linewidth=1.5,
                            alpha=0.7)
            ax2.set_ylabel("θ (Predation Control)", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(-0.05, 1.05)  # Fixed scale for theta across all subplots
            
            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right', fontsize=8)
        
        plt.tight_layout()
    
    st.pyplot(fig)
    st.caption("Figure 1: Population dynamics under different constant θ values (see dashed line for value)")

    st.markdown(r"""
    As we can see from the charts using the default parameters, the system rapidly reaches the Allee threshold and collapses when θ is set to 1.0, which is behavior identical to
    what we would expect from the system we examined from Equation (2). In fact, there are many values of $\theta$ for which this is true. However, with our base parameters all set to 1.0, our starting conditions
    of 10 sheep and 10 wolves, a theta value of 0.2 is sufficient to "rescue" the system from collapse below the Allee threshold.

    Since we are using simple default values, it is easy to explore how the combinbation of $\theta$ and $\beta$ multiply to yield what is an effective $\beta$, or $\beta_{eff}$, for the system.

    To see this, try setting $\beta$ in the sidebar to the value 0.2. We will then see that the plot is updated so that the $\theta$ = 1.0 pane is identical to the what we saw in the 0.2 pane with the default settings. The effects
    on all four panes should be instructive.

    As we can see, a modulation of predation intensity can have a stabilizing effect on the system even when it is subject to an Allee threshold. However,
    if were not interested in control, we could achieve this simply through adjusting the $\beta$ parameter alone. In our next example, we will assign
    functional control to the $\theta$ parameter in a manner that distinguishes the parameter as a controlling variable.
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
