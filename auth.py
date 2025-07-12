# auth.py

import streamlit as st
from auth_utils import login_user, signup_user

def login_form():
    st.title("🔐 Login to NotebookLM")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("📧 Email")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login"):
            success, msg = login_user(email, password)
            if success:
                st.success(msg)
                st.experimental_rerun()
            else:
                st.error(msg)

    with tab2:
        email = st.text_input("📧 Email", key="signup_email")
        password = st.text_input("🔑 Password", type="password", key="signup_pw")
        if st.button("Sign Up"):
            success, msg = signup_user(email, password)
            if success:
                st.success(msg)
                st.experimental_rerun()
            else:
                st.error(msg)
