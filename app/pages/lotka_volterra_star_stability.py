import streamlit as st

#--- Streamlit page building ---#
def render_sidebar() -> None:
    pass

def render_intro() -> None:
    st.markdown("## LV* with stability analysis")

def render_footer() -> None:
    st.divider()
    st.markdown("### Quick Navigation")
    st.page_link("pages/home.py", label="Home", icon="🏠")


def main() -> None:
    render_sidebar()
    render_intro()
    render_footer()

main()