import streamlit as st


# --- Streamlit page building ---#
def render_sidebar() -> None:
    st.sidebar.header("Hello!")


def render_intro() -> None:
    st.markdown(
        r"""## Home
This "book" is provided as a walkthrough from a standard Lotka-Volterra model to a modified version of interest for adding a control variable.
    The modified version, or at least the *primary* modified version, is a mostly standard model with the additions of a carrying capacity for the prey and an adjustable integer minimum for both species.
    We then include an additional variable to that modified version for predator behavior control.

We refer to the primary modified version of Lotka-Volterra as LV* or $LV^*$ in this book. Where applicable, we refer to the further modifications of that
    system as LV* with $\theta$ as a control variable or LV* with adaptive $\theta$ as a control variable.
    """
    )

    st.markdown("""### Table of Symbols""")

    st.markdown(
        r"""
    The following table lists all symbols used throughout the interactive demonstrations:
    
    | Symbol | Description | Typical Range |
    |--------|-------------|---------------|
    | $s$ | Sheep (prey) population density | $s \geq 0$ |
    | $w$ | Wolves (predator) population density | $w \geq 0$ |
    | $\alpha$ | Prey intrinsic growth rate | $\alpha > 0$ |
    | $\beta$ | Predation rate coefficient (attack rate) | $\beta > 0$ |
    | $\gamma$ | Predator death rate | $\gamma > 0$ |
    | $\delta$ | Predator growth efficiency from predation | $\delta > 0$ |
    | $K$ | Carrying capacity for prey population | $K > 0$ |
    | $A$ | Allee threshold (minimum viable population) | $A \geq 0$ |
    | $\theta$ | Predation control parameter (modulation factor) | $\theta \in [0, 1]$ |
    | $k$ | Sensitivity parameter for adaptive theta control | $k \in [0, 1]$ |
    | $\varepsilon$ | Small constant to prevent division by zero | $\varepsilon \approx 10^{-8}$ |
    | $t$ | Time | $t \geq 0$ |
    
    **Notes:**
    - In the base Lotka-Volterra system (Equation 1), only $s$, $w$, $\alpha$, $\beta$, $\gamma$, and $\delta$ are used.
    - LV* adds the carrying capacity $K$ and Allee threshold $A$.
    - LV* with control introduces $\theta$ as a modulation parameter.
    - Adaptive control uses $k$ to define $\theta(s; k, K) = \frac{s + \varepsilon}{(s + \varepsilon) + kK}$.
    - When $\theta = 1$, the system reduces to standard LV*. When $\theta = 0$, predation is completely suppressed.
    """
    )


def main() -> None:
    render_sidebar()
    render_intro()


main()
