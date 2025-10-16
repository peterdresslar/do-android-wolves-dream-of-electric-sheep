import streamlit as st
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
from utils.lv_core import lv_star_ode_ivp



#--- Analysis functions ---#


#--- Streamlit page building ---#
def render_sidebar() -> None:
    if st.sidebar.button("Clear cache and recompute"):
        st.cache_data.clear()
        st.rerun()

def render_intro() -> None:
    st.markdown("## LV* with $\\theta$ as a control variable")

def render_analysis() -> None:
    st.markdown("""
    Here, we explore the addition of a controlling parameter called $\theta$ to the LV* system. $\theta$ is a parameter that controls the strength of the predation on the sheep population.
    The system is given by the following differential equations:
    $$
    \begin{align}
    \frac{ds}{dt} &= \alpha s - \theta \beta s w \\
    \frac{dw}{dt} &= - \gamma w + \delta \theta \beta s w
    \end{align}
    $$
    """)

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