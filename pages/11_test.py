import getpass
import os
import streamlit as st

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.text_input("OpenAI API key", type='password')

st.write("API Key : " + os.environ["OPENAI_API_KEY"])
