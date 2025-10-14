import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="Do Android Wolves Dream of Electric Sheep?",
        page_icon="�",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def render_sidebar() -> None:
    pass
        

def render_intro() -> None:
    st.markdown(r"""
    ### Tracing from Volterra to our implementation
    We acknowledge that Diz-Pita & Otero-Espinar present the predator–prey system in a way that faithfully updates Volterra's original two-step development, using the factored-rate form:
    $$
    \dot x = x\,(a - b\,y),\qquad \dot y = y\,(-c + d\,x).
    $$
    This is all but symbolically equivalent to Volterra's own notation (rates inside parentheses multiplying the state).

    For functional purposes, we expand the two equations, which is algebraically equivalent and more convenient to implement as computer code:
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

def render_example() -> None:
    st.markdown("""
    ### Example: Base Lotka-Volterra
    
    """)

def render_example_2() -> None:
    st.markdown("""
    ### Example: Lotka-Volterra with a carrying capacity $K$
    
    """)

def main() -> None:
    configure_page()
    render_sidebar()
    render_intro()
    render_example()
    render_example_2()

if __name__ == "__main__":
    main()