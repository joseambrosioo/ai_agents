import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file (optional, but good practice)
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please set the GOOGLE_API_KEY environment variable or add it to a .env file.")
else:
    genai.configure(api_key=api_key)

    # --- Streamlit App ---
    st.set_page_config(page_title="AI Chatbot Assistant", page_icon="💬")
    st.title("💬 AI Chatbot Assistant")
    st.write("Interact with your AI assistant.")

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for the user
    if prompt := st.chat_input("What's up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare the chat history for the model
        # The Gemini API expects messages in a specific format (role: user/model, parts: text)
        model_messages = []
        for msg in st.session_state.messages:
            # Ensure roles are 'user' or 'model'
            role = 'user' if msg['role'] == 'user' else 'model'
            model_messages.append({'role': role, 'parts': [{'text': msg['content']}]})

        try:
            # Initialize the generative model
            # Use the 'gemini-2.0-flash-exp' model
            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            # Start a chat session with the existing history
            chat_session = model.start_chat(history=model_messages)

            # Send the user's latest message to the model
            # Note: The latest user message is already in model_messages,
            # start_chat handles sending the last message when history is provided.
            # We just need to get the response.
            response = chat_session.send_message(prompt)


            # Add model response to chat history
            msg_content = response.text
            st.session_state.messages.append({"role": "assistant", "content": msg_content})
            # Display model response
            with st.chat_message("assistant"):
                st.markdown(msg_content)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Optionally remove the last user message if the API call failed
            # st.session_state.messages.pop()


    st.markdown("---")
    st.markdown("© Copyright 2025, created by Jose Ambrosio")
