import streamlit as st

home_page = st.Page("pages/home.py", title="Application Home")
lv_page = st.Page("pages/lotka_volterra.py", title="Base Lotka-Volterra and a modified version with simple thresholds")
lv_star_page = st.Page("pages/lotka_volterra_star_stability.py", title="LV* stability analysis")
lv_star_with_theta_page = st.Page("pages/lv_star_with_theta.py", title="LV* with theta as a control variable")

def main() -> None:
    st.set_page_config(
        page_title="Do Android Wolves Dream of Electric Sheep?",
        page_icon="🐺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    nav = st.navigation([home_page, lv_page, lv_star_page, lv_star_with_theta_page])
    nav.run()

if __name__ == "__main__":
    main()