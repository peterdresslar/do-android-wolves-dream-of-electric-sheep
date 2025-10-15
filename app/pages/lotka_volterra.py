import streamlit as st
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
import altair as alt

# Constant DT: since our solver parameters are hardcoded it makes sense to also use a constant DT
DT = 0.02
# the following are solve_ivp tuning parameters
ATOL = 1e-8
RTOL = 1e-8
DENSE_OUTPUT = True

#--- ODEs and helpers ---#
def base_lv_ode(s: float, w: float, alpha: float, beta: float, gamma: float, delta: float) -> tuple[float, float]:
    ds_dt = alpha * s - beta * s * w
    dw_dt = -gamma * w + delta * s * w
    return ds_dt, dw_dt

def lv_star_ode(s: float, w: float, alpha: float, beta: float, gamma: float, delta: float, K: float, A: float) -> tuple[float, float]:
    if s >= K:
        ds_dt = min(0.0, alpha * s - beta * s * w)  # at capacity we can shrink but not grow
                                                    # note that alpha is still here since we might get a decline dampened by growth even at the cap.
    else:
        ds_dt = alpha * s - beta * s * w

    dw_dt = -gamma * w + delta * s * w
    return ds_dt, dw_dt

def base_lv_ode_ivp(t: float, x: tuple[float, float], alpha: float, beta: float, gamma: float, delta: float) -> tuple[float, float]:
    s, w = x
    return base_lv_ode(s, w, alpha, beta, gamma, delta)

def lv_star_ode_ivp(t: float, x: tuple[float, float], alpha: float, beta: float, gamma: float, delta: float, K: float, A: float) -> tuple[float, float]:
    s, w = x
    return lv_star_ode(s, w, alpha, beta, gamma, delta, K, A)

def allee_sheep_event(t: float, x: tuple[float, float], alpha: float, beta: float, gamma: float, delta: float, K: float, A: float) -> float:
    return x[0] - A

def allee_wolf_event(t: float, x: tuple[float, float], alpha: float, beta: float, gamma: float, delta: float, K: float, A: float) -> float:
    return x[1] - A

def allee_terminal_event(t: float, x: tuple[float, float], alpha: float, beta: float, gamma: float, delta: float, K: float, A: float) -> float:
    return max(x[0] - A, x[1] - A)

def reset_events() -> None:
    allee_sheep_event.terminal = False
    allee_wolf_event.terminal = False
    allee_terminal_event.terminal = True

def calculate_phase_space_vectors(alpha: float, beta: float, gamma: float, delta: float,
                                  s0: float, w0: float,
                                  t_max: float = 50.0) -> np.ndarray:  # removed dt parameter
    reset_events()

    t_eval = np.arange(0.0, t_max, DT)  # use global DT
    sol = solve_ivp(
        base_lv_ode_ivp,
        [0.0, t_max],
        [s0, w0],
        args=(alpha, beta, gamma, delta),
        method="RK45",
        t_eval=t_eval,
        rtol=RTOL,
        atol=ATOL,
        dense_output=DENSE_OUTPUT
    )
    s = sol.y[0]
    w = sol.y[1]
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    return np.column_stack([s, w])

#--- Streamlit helper ---#
def reset_to_defaults() -> None:
    st.session_state.alpha = 1.0
    st.session_state.beta = 0.1
    st.session_state.gamma = 1.5
    st.session_state.delta = 0.75
    st.session_state.s_start = 10
    st.session_state.w_start = 10
    st.session_state.T = 50.0
    st.session_state.K = 1000
    st.session_state.A = 2

def set_preset_2a() -> None:
    # this will be rescued by the K cap
    #1, .05, 1, .05, 75, 2, 100, 2
    st.session_state.alpha = 1.0
    st.session_state.beta = 0.05
    st.session_state.gamma = 1.0
    st.session_state.delta = 0.05
    st.session_state.s_start = 75
    st.session_state.w_start = 2
    st.session_state.K = 100
    st.session_state.A = 2

def set_preset_2b() -> None:
    # this will NOT be rescued by the higher K cap
    #1, .05, 1, .05, 75, 2, 200, 2
    st.session_state.alpha = 1.0
    st.session_state.beta = 0.05
    st.session_state.gamma = 1.0
    st.session_state.delta = 0.05
    st.session_state.s_start = 75
    st.session_state.w_start = 2
    st.session_state.K = 200
    st.session_state.A = 2

#--- Streamlit page building ---#
def render_intro() -> None:
    st.markdown(r"""
    ### Tracing from Volterra to our implementation
    The Lotka-Volterra system comprises two ordinary differential equations (ODEs) that describe the dynamics of two interacting species in which one species (predators) consumes members of the other species (prey).
    We acknowledge that Diz-Pita & Otero-Espinar present the system in a way that faithfully updates Prof. Volterra's original two-step development, using the factored-rate form:
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

def add_example_2_sidebar() -> None:
    # group for example 2. just carrying capacity K
    st.sidebar.markdown("### Example 2")
    st.sidebar.number_input("K", key="K", value=200, min_value=0, max_value=1000, step=1)
    st.sidebar.number_input("A", key="A", value=2, min_value=0, max_value=10, step=1)
    st.sidebar.button("Preset 2a", on_click=set_preset_2a)
    st.sidebar.button("Preset 2b", on_click=set_preset_2b)

def render_example_1() -> None:
    st.markdown("""
    ### Example 1: Base Lotka-Volterra
    As we can see from the equations above, we have four parameters and two initial conditions. When we apply fixed positive values to the parameters and pick positive values for the initial conditions, we will
    have a deterministic outcome for the system at any time point $t$. This outcome can be computed by integrating the ODEs of the system (1) for that time point to supply values for states $s(t)$ and $w(t)$.
    """)
    reset_events()
    T = st.session_state.T
    alpha = st.session_state.alpha
    beta = st.session_state.beta
    gamma = st.session_state.gamma
    delta = st.session_state.delta
    s_start = st.session_state.s_start
    w_start = st.session_state.w_start

    t_eval = np.arange(0.0, T, DT)

    solution = solve_ivp(base_lv_ode_ivp,                  # base system of equations
                        [0.0, T],                          # time range
                        [s_start, w_start],                # initial conditions
                        t_eval=t_eval,                     # time points to evaluate the solution at 
                        args=(alpha, beta, gamma, delta),  # parameters
                        method="RK45",                     # solver
                        rtol=RTOL,                         
                        atol=ATOL,                         
                        dense_output=DENSE_OUTPUT          
                        # here we do not need events, they are irrelevant to base LV
                        )
    # add labels to the solution arrays by converting solution.y.T to a pandas dataframe and adding a columns field
    solution_df = pd.DataFrame(solution.y.T, columns=["Sheep", "Wolves"])
    st.caption("Figure 1")
    st.line_chart(solution_df, x_label="Time", y_label="Population Density") # plot the solution
    
    st.markdown("""
    You can use the Example 1 sidebar controls to change the parameters and initial conditions and see the effect on the solution above. With
    experimentation we can see that the system oscillates in a periodic fashion with any positive inputs for the parameters. In fact, by switching our view to
    a phase space plot, where we simply compare the sheep and wolf outcomes, we can see this closed oscillation more clearly.
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
    st.caption("Figure 2")
    s_null = alt.Chart(pd.DataFrame({'x': [gamma / delta]})).mark_rule(color='red', strokeDash=[4, 4]).encode(x='x:Q')
    w_null = alt.Chart(pd.DataFrame({'y': [alpha / beta]})).mark_rule(color='blue', strokeDash=[4, 4]).encode(y='y:Q')
    eq_pt = alt.Chart(pd.DataFrame({'Sheep': [gamma / delta], 'Wolves': [alpha / beta]})).mark_point(color='black', size=60)

    st.altair_chart(phase_chart + s_null + w_null + eq_pt, use_container_width=True)
    st.caption("Dashed lines: nullclines; dot: coexistence equilibrium (γ/δ, α/β).")
        


    
def render_example_2() -> None:
    st.markdown(r"""
    ### Example 2: LV*: Lotka-Volterra with a carrying capacity $K$ and simple Allee effect
    Early modifications to the Lotka-Volterra system were made to account for the fact that the prey population cannot grow indefinitely. The carrying capacity $K$ was introduced to limit the maximum population size of the prey in particular,
    which in turn limits the effect maximum population size of the predators.

    The Allee effect was an early extension of the Lotka-Volterra system devised to account for the fact that the prey population cannot grow indefinitely. The Allee effect is a parameter that limits the minimum population size of the prey.

    Here we implement both of these effects in our system in a simple manner. We add a carrying capacity $K$ to as an absolute limit on the prey growth function and a simple Allee threshold value $A$, below which either the prey population or the predator population will crash to zero.
    Describing these in terms of a system of equations for our LV*, we have:

    $$
    \begin{aligned}
    \dot s &= \alpha s - \beta sw \\
    \dot w &= -\gamma w + \delta sw
    \end{aligned}
    \tag{2}
    $$

    subject to:

    $$
    \begin{cases}
    s(t) = \min(s(t), K) \\[0.5em]
    s(t) = 0 & \text{if } s(t) < A \\[0.5em]
    w(t) = 0 & \text{if } w(t) < A
    \end{cases}
    \tag{2a}
    $$
    """)
    reset_events()
    T = st.session_state.T
    A = st.session_state.A
    K = st.session_state.K
    s_start = st.session_state.s_start
    w_start = st.session_state.w_start
    alpha = st.session_state.alpha
    beta = st.session_state.beta
    gamma = st.session_state.gamma
    delta = st.session_state.delta

    t_eval = np.arange(0.0, T, DT)

    steps = int(round(T / DT)) # for use in the chart

    lv_star_solution = solve_ivp(lv_star_ode_ivp,
                        [0.0, T],
                        [s_start, w_start],
                        args=(alpha, beta, gamma, delta, K, A),
                        t_eval=t_eval,
                        method="RK45",
                        rtol=RTOL,
                        atol=ATOL,
                        dense_output=DENSE_OUTPUT,
                        events=[allee_sheep_event, allee_wolf_event, allee_terminal_event]  # events that change system trajectory
                        )

    # post process: we do want the chart to show the entire time range generated from T even though our arrays stop.
    # this makes it easier to compare with the base LV solution.
    lv_star_solution_df = pd.DataFrame(
        lv_star_solution.y.T,
        index=lv_star_solution.t,
        columns=["Sheep", "Wolves"],
    ).reindex(t_eval).fillna(0.0)

    # match Example 1's x-axis (sample index 0..T/DT)
    lv_star_solution_df.index = np.arange(len(t_eval))

    st.caption("Figure 3")
    st.line_chart(lv_star_solution_df, x_label="Time", y_label="Population Density") # plot the solution
        
    st.markdown("""
    Here we can see that the modified LV* system exhibits a new possibility: the system can crash to zero as a result of one or the other species declining beneath the Allee threshold. 
    Using the sidebar controls for both the example 1 parameters and initial conditions as well as the example 2 modifying thresholds, it is possible to evaluate settings that lead to stability,
    and ones that lead to system collapse.

    Please note that in this diagram we simply stop processing once either the sheep or the wolves crash to zero. The real dynamics would have the other species following a trajectory to a certain eventual crash (wolves) or explosion (sheep).
    
    Below we can see our phase portrait for the LV* system given the parameters and initial conditions we selected. Unsurprisingly, system dynamics that crash below the A threshold clip into irretrievable states in the phase portrait.
    """)

    phase_df = pd.DataFrame({
        "t": lv_star_solution.t,
        "Sheep": lv_star_solution.y[0],
        "Wolves": lv_star_solution.y[1],
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
    st.caption("Figure 4")
    s_cap = alt.Chart(pd.DataFrame({'x': [K]})).mark_rule(color='orange').encode(x='x:Q')
    allee_s = alt.Chart(pd.DataFrame({'x': [A]})).mark_rule(color='purple', strokeDash=[2, 2]).encode(x='x:Q')
    allee_w = alt.Chart(pd.DataFrame({'y': [A]})).mark_rule(color='purple', strokeDash=[2, 2]).encode(y='y:Q')

    st.altair_chart(phase_chart + s_cap + allee_s + allee_w, use_container_width=True)
    st.caption("Orange: carrying capacity K; purple dashed: Allee thresholds A.")

    st.markdown("""
    There are two presets for example 2 that demonstrate the effects of the K cap and the A threshold. Preset 2a will be rescued by the K cap, while preset 2b will not.
    The difference between the two presets is the value of the carrying capacity K. Preset 2a has a K of 100, while preset 2b has a K of 200. Perhaps somewhat paradoxically,
    a lower K cap can lead to a more stable modified system in the presence of an Allee threshold. This counterintuitive behavior connects to the well-known
    “paradox of enrichment” in predator–prey systems (Rosenzweig, 1971), where increasing carrying capacity can destabilize dynamics and heighten collapse risk.
    """)

def render_footer() -> None:
    st.divider()
    st.markdown("### Quick Navigation")
    st.page_link("pages/lotka_volterra_star_stability.py", label="Next: LV* with stability analysis", icon="➡️")
    st.page_link("pages/home.py", label="Home", icon="🏠")

def main() -> None:
    add_sidebar()
    render_intro()
    add_example_1_sidebar()
    add_example_2_sidebar()
    render_example_1()
    render_example_2()
    render_footer()

main()