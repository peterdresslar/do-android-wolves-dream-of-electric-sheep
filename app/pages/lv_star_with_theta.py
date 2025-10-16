import streamlit as st
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
from utils.lv_core import lv_star_with_theta_ode_ivp

# Constant DT: since our solver parameters are hardcoded it makes sense to also use a constant DT
DT = 0.02
# the following are solve_ivp tuning parameters
ATOL = 1e-8
RTOL = 1e-8
DENSE_OUTPUT = True

#--- Analysis functions ---#


#--- Streamlit helper ---#
def reset_to_defaults() -> None:
    st.session_state.Time = 50.0
    st.session_state.s_start = 10
    st.session_state.w_start = 10
    st.session_state.alpha = 1.0
    st.session_state.beta = 0.1
    st.session_state.gamma = 1.5
    st.session_state.delta = 0.75

#--- Streamlit page building ---#
def add_sidebar() -> None:
    st.sidebar.header("Controls")
    st.sidebar.button("Reset to defaults", on_click=reset_to_defaults)
    st.sidebar.slider('Time', key="T", value=50.0, min_value=1.0, max_value=250.0, step=1.0)

def add_example_1_sidebar() -> None:
    # group for example 1. the four params and two initial conditions
    # section title for sidebar
    st.sidebar.markdown("### Example 1")
    st.sidebar.number_input("alpha", key="alpha", value=1.0, min_value=0.0, max_value=10.0, step=0.1)
    st.sidebar.number_input("beta", key="beta", value=0.1, min_value=0.0, max_value=10.0, step=0.1)
    st.sidebar.number_input("gamma", key="gamma", value=1.5, min_value=0.0, max_value=10.0, step=0.1)
    st.sidebar.number_input("delta", key="delta", value=0.75, min_value=0.0, max_value=10.0, step=0.1)
    st.sidebar.number_input("s_start", key="s_start", value=10, min_value=0, max_value=1000, step=1)
    st.sidebar.number_input("w_start", key="w_start", value=10, min_value=0, max_value=1000, step=1)


def render_intro() -> None:
    st.markdown("## LV* with $\\theta$ as a control variable")

def render_analysis() -> None:
    st.markdown(r"""
    We have observed that the stable oscillations of the standard Lotka-Volterra dynamical system are disrupted in a wide range of cases by the introduction of a simple but biologically plausible Allee threshold condition. 
    In real life population studies, however, the system successfully predicts population pair outcomes in a variety of ecologies, and with diverse parameter settings.
    Further, in Indvidual Based Modeling (IBM) techniques where the population is transformed from a single census variable into integerized unitary members, operations on a population below
    the specific threshold of two (or, taking things to a narrow extreme: one) seem particularly dischordant with respect to biological reality. Thus, we might be interested in ways
    to control the operations of the system in a manner that avoids the extinction threshold. 

    These motications lead us to we explore the addition of a controlling parameter called $\theta$ to the LV* system. 
    
    In our implementation, we materialize an experimental system in which the predators are able (or compelled) to adjust their predation intensity in response to environmental factors. Relevant examples exist in the literature both 
    in theoretical models and in real-world observations. This control can be applied in the form of a single parameter, $\theta$, that modulates predation intensity $\beta$ in both the equations of the modified system.

    The system is given by the following differential equations: [TODO NUMBER]
    $$
    \begin{align}
    \frac{ds}{dt} &= \alpha s - \theta \beta s w \\
    \frac{dw}{dt} &= - \gamma w + \delta \theta \beta s w
    \end{align} \tag{3}
    $$

    Where $\theta \in [0, 1]$. 
    
    Observe that the equations in (3) imply that where $\theta$ is set to 0, predation is completely dampend; yielding a $\beta_eff$ of 0, with the resultant effects on both populations. When $\theta$ is set to 1, the operation of
    the system is identical to the LV* system with no control.

    Let's consider the "operation" of the control through different methods. Of course, the most simple of these methods would be to set $\theta$ to a constant value. This would be the equivalent of a "programmed" variable, in the sense that the value of $\theta$ is set before the simulation begins.
    """)

def render_example_one() -> None:
    """
    Four charts with constant theta values of 0, 0.2, 0.5, and 1.0.
    """

    T = st.session_state.Time
    theta_values = [0.0, 0.2, 0.5, 1.0]
    for theta in theta_values:
        solution = solve_ivp(lv_star_with_theta_ode_ivp, [0.0, st.session_state.Time], [st.session_state.s_start, st.session_state.w_start], args=(st.session_state.alpha, st.session_state.beta, st.session_state.gamma, st.session_state.delta, st.session_state.K, st.session_state.A, theta), method="RK45", t_eval=np.arange(0.0, st.session_state.Time, DT), rtol=RTOL, atol=ATOL, dense_output=DENSE_OUTPUT)
        st.line_chart(solution.y.T, x_label="Time", y_label="Population Density")
    

def render_footer() -> None:
    st.divider()
    st.markdown("### Quick Navigation")
    st.page_link("pages/lv_star_plus_theta.py", label="Next: LV* with theta as a programmed variable", icon="➡️")
    st.page_link("pages/home.py", label="Home", icon="🏠")


def main() -> None:
    render_sidebar()
    render_intro()
    render_analysis()
    render_footer()

main()