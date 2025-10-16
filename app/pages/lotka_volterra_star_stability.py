import streamlit as st
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
from utils.lv_core import lv_star_ode_ivp

T = 50.0
DT = 0.02

RTOL = 1e-8
ATOL = 1e-8
DENSE_OUTPUT = False

#--- Four presets for analysis ---#
def set_preset_1() -> tuple[float, float, float, float, float, float]:
    alpha = 1.0
    beta = 0.1
    gamma = 1.5
    delta = 0.75
    K = 20
    A = 1
    return alpha, beta, gamma, delta, K, A

def set_preset_2() -> tuple[float, float, float, float, float, float]:
    alpha = 1.0
    beta = 0.1
    gamma = 1.0
    delta = 0.1
    K = 50
    A = 1
    return alpha, beta, gamma, delta, K, A

def set_preset_3() -> tuple[float, float, float, float, float, float]:
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    delta = 1.0
    K = 20
    A = 1
    return alpha, beta, gamma, delta, K, A

def set_preset_4() -> tuple[float, float, float, float, float, float]:
    alpha = 0.8
    beta = 0.3
    gamma = 0.8
    delta = 0.1
    K = 20
    A = 1
    return alpha, beta, gamma, delta, K, A


#--- Analysis functions ---#
def compute_time_to_crash(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    K: float,
    A: float,
    s_start: float,
    w_start: float,
) -> float | None:
    t_eval = np.arange(0.0, T, DT)
    sol = solve_ivp(
        lv_star_ode_ivp,
        [0.0, T],
        [s_start, w_start],
        args=(alpha, beta, gamma, delta, K, A),
        method="RK45",
        t_eval=t_eval,
        rtol=RTOL,
        atol=ATOL,
        dense_output=DENSE_OUTPUT,
    )
    s = sol.y[0]
    w = sol.y[1]
    below = np.where((s < A) | (w < A))[0]
    if below.size == 0:
        return None
    idx = int(below[0])
    return float(t_eval[idx])

@st.cache_data(show_spinner=False, max_entries=16)
def compute_stability_surface(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    K: float,
    A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_vals = np.arange(A, K, 2)
    w_vals = np.arange(A, K, 2)
    S, W = np.meshgrid(s_vals, w_vals, indexing="ij")
    Z = np.zeros_like(S, dtype=float)

    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            t_crash = compute_time_to_crash(
                alpha, beta, gamma, delta, K, A, float(S[i, j]), float(W[i, j])
            )
            if t_crash is None:
                Z[i, j] = T
            else:
                Z[i, j] = t_crash

    return S, W, Z



#--- Streamlit page building ---#
def render_sidebar() -> None:
    if st.sidebar.button("Clear cache and recompute"):
        st.cache_data.clear()
        st.rerun()

def render_intro() -> None:
    st.markdown("## LV* stability analysis")

def render_analysis() -> None:
    st.markdown("""
    Here, we plot the stability surface for the LV* system. The z-axis is the time to crash, and the x- and y-axes are the initial conditions for the sheep and wolves respectively. 
    The color of the surface represents the time to crash. The maximum time to crash is T, which is the horizon of the system.
    Thus, maximum plot values indicate that there was no crash.

    We investigate 4 various presets for the parameters of the LV* system. Preset 2 is given a large carrying capacity in order to
    plot the larger stability surface for that preset.
    """)

    presets = [
        ("Preset 1", set_preset_1),
        ("Preset 2", set_preset_2),
        ("Preset 3", set_preset_3),
        ("Preset 4", set_preset_4),
    ]

    for i, (title, fn) in enumerate(presets):
        st.caption(f"Figure {i + 1}")
        alpha, beta, gamma, delta, K, A = fn()
        with st.spinner(f"Computing {title} surface..."):
            S, W, Z = compute_stability_surface(alpha, beta, gamma, delta, K, A)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(S, W, Z, cmap="viridis", edgecolor="none", antialiased=True)
        ax.set_title(title)
        ax.set_xlabel("Sheep start (s0)")
        ax.set_ylabel("Wolves start (w0)")
        ax.set_zlabel("t_crash (Max: no crash)")
        ax.set_zlim(0, T)
        fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)
        st.pyplot(fig, clear_figure=True)
        st.markdown(fr"**Preset:** $\alpha = {alpha}, \beta = {beta}, \gamma = {gamma}, \delta = {delta}, K = {K}, A = {A}$")

def render_footer() -> None:
    st.divider()
    st.markdown("### Quick Navigation")
    st.page_link("pages/lv_star_with_theta.py", label="Next: LV* with theta as a programmed variable", icon="➡️")
    st.page_link("pages/home.py", label="Home", icon="🏠")


def main() -> None:
    render_sidebar()
    render_intro()
    render_analysis()
    render_footer()

main()