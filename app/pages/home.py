import streamlit as st

#--- Streamlit page building ---#
def render_sidebar() -> None:
    pass

def render_intro() -> None:
    st.markdown("## Home")

def main() -> None:
    render_sidebar()
    render_intro()
        
main()