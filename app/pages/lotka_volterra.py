import streamlit as st
from scipy.integrate import solve_ivp
import pandas as pd

def base_lv_ode(s: float, w: float, alpha: float, beta: float, gamma: float, delta: float) -> tuple[float, float]:
    ds_dt = alpha * s - beta * s * w
    dw_dt = -gamma * w + delta * s * w
    return ds_dt, dw_dt

def base_lv_ode_ivp(t: float, x: tuple[float, float], alpha: float, beta: float, gamma: float, delta: float) -> tuple[float, float]:
    s, w = x
    return base_lv_ode(s, w, alpha, beta, gamma, delta)



def configure_page() -> None:
    st.set_page_config(
        page_title="Do Android Wolves Dream of Electric Sheep?",
        page_icon="🐺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def render_sidebar() -> None:
    pass
        

def render_intro() -> None:
    st.markdown(r"""
    ### Tracing from Volterra to our implementation
    The Lotka-Volterra system comprises two ordinary differential equations (ODEs) that describe the dynamics of two interacting species in which one species (predators) consumes members of the other species (prey).
    We acknowledge that Diz-Pita & Otero-Espinar present the system in a way that faithfully updates Volterra's original two-step development, using the factored-rate form:
    $$
    \dot x = x\,(a - b\,y),\qquad \dot y = y\,(-c + d\,x).
    $$
    This is all but symbolically equivalent to Volterra's own notation (rates inside parentheses multiplying the state).

    For implementation purposes, we expand the two equations, which is algebraically equivalent and more convenient to implement as computer code:
    $$
    \dot x = a\,x - b\,x\,y,\qquad \dot y = -c\,y + d\,x\,y.
    $$

    In our manuscript and code we adopt the conventional Greek parameters and species names:
    $$
    (a,b,c,d)\ \mapsto\ (\alpha,\beta,\gamma,\delta),\qquad
    (x,y)\ \mapsto\ (s\ \text{(sheep)},\ w\ \text{(wolves)}).
    $$

    Thus our implementation is of the following equations:
    $$
    \begin{aligned}
    \dot s &= \alpha s - \beta sw \\
    \dot w &= -\gamma w + \delta sw
    \end{aligned}
    \tag{1}
    $$
    """)

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

def add_example_2_sidebar() -> None:
    # group for example 2. just carrying capacity K
    st.sidebar.markdown("### Example 2")
    st.sidebar.number_input("K", value=1000, min_value=0, max_value=1000, step=1)

def render_example_1() -> None:
    st.markdown("""
    ### Example 1: Base Lotka-Volterra
    As we can see from the equations above, we have four parameters and two initial conditions. When we apply fixed positive values to the parameters and pick positive values for the initial conditions, we will
    have a deterministic outcome for the system at any time point $t$. This outcome can be computed by integrating the ODEs of the system (1) for that time point to supply values for states $s(t)$ and $w(t)$.
    """)
    alpha = st.session_state.alpha
    beta = st.session_state.beta
    gamma = st.session_state.gamma
    delta = st.session_state.delta
    s_start = st.session_state.s_start
    w_start = st.session_state.w_start
    solution = solve_ivp(base_lv_ode_ivp,                  # base system of equations
                        [0.0, 50.0],                       # time range
                        [s_start, w_start],                # initial conditions
                        first_step=0.02,                   # first step size
                        max_step=0.02,                     # max step size
                        args=(alpha, beta, gamma, delta),  # parameters
                        method="RK45",                     # solver
                        rtol=1e-8,                         # relative tolerance
                        atol=1e-8)                         # absolute tolerance
    # add labels to the solution arrays by converting solution.y.T to a pandas dataframe and adding a columns field
    solution_df = pd.DataFrame(solution.y.T, columns=["Sheep", "Wolves"])
    st.line_chart(solution_df, x_label="Time", y_label="Population Density") # plot the solution
    
    st.markdown("""
    You can use the Example 1 sidebar controls to change the parameters and initial conditions and see the effect on the solution above. With
    experimentation we can see that the system oscillates in a periodic fashion with any positive inputs for the parameters. In fact, by switching our view to
    a phase space plot, where we simply compare the sheep and wolf outcomes, we can see this closed oscillation more clearly.
    """)


    
def render_example_2() -> None:
    st.markdown("""
    ### Example 2: Lotka-Volterra with a carrying capacity $K$
    
    """)

def main() -> None:
    configure_page()
    render_sidebar()
    render_intro()
    add_example_1_sidebar()
    add_example_2_sidebar()
    render_example_1()
    render_example_2()

if __name__ == "__main__":
    main()