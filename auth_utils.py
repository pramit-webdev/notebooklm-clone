# auth_utils.py

import streamlit as st
from supabase_client import supabase

def login_user(email, password):
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            st.session_state["user"] = {
                "id": response.user.id,
                "email": response.user.email,
                "access_token": response.session.access_token
            }
            return True, "Login successful"
        return False, "Login failed"
    except Exception as e:
        return False, str(e)

def signup_user(email, password):
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            st.session_state["user"] = {
                "id": response.user.id,
                "email": response.user.email,
                "access_token": response.session.access_token
            }
            return True, "Signup successful"
        return False, "Signup failed"
    except Exception as e:
        return False, str(e)

def logout_user():
    if "user" in st.session_state:
        st.session_state.pop("user")
