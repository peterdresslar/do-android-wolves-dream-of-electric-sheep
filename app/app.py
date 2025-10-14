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
        

def render_home_page() -> None:
    st.write("## Home")

def main() -> None:
    configure_page()
    render_sidebar()
    render_home_page()

if __name__ == "__main__":
    main()