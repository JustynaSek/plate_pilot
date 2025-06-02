# src/test_app.py
import streamlit as st
import os

print("--- DEBUG: Test app starting ---") # Extremely early print

st.set_page_config(layout="wide")
st.title("Test Streamlit App is Running!")
st.write("If you see this, the basic environment works.")
st.write(f"Python version: {os.sys.version}")
st.write(f"Streamlit version: {st.__version__}")
print("--- DEBUG: Test app finished UI setup ---")