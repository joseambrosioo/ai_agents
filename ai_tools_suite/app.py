# my_streamlit_app/streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="Multi AI Agents App",
    page_icon="🤖",
)

st.title("Welcome to the Multi AI Agents App!")

st.markdown(
    """
    Use the menu on the left to navigate between the different AI tools:
    
    - **AI Chatbot:** Interact with a conversational AI assistant.
    - **Multi-File Analyzer:** Upload and analyze various document types and URLs with AI.
    - **AI Blog Post Generator:** Use AI to write and generate blog content.
    
    ---
    """
)

st.info("Select an application from the sidebar menu to begin.")

st.markdown("---")
st.markdown("© Copyright 2025, created by Jose Ambrosio")