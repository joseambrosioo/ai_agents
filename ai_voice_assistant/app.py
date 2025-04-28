#import necessary libraries
import os
import base64
import streamlit as st      
from audio_recorder_streamlit import audio_recorder
from google import genai
from google.genai import types
import openai
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# initialize the Google Gemini client 
# initialize the OpenAI client  
def setup_openai_client(api_key):
    return openai.OpenAI(api_key=OPENAI_API_KEY)

# function to transcribe audio to text using OpenAI's Whisper API
# def transcribe_audio_with_openai(client, audio_path):
#     with open(audio_path, "rb") as audio_file:
#         transcript = client.Audio.transcribe(
#             model="whisper-1",
#             file=audio_file,
#             response_format="text"
#         )
#     return transcript["text"]

# function to transcribe audio to text using OpenAI's Whisper API
def transcribe_audio_with_openai(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript.text

# taking response from OpenAI API
def fetch_ai_response(client, input_text):
    message = [{"role": "user", "content": input_text}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message["content"]

# convert text to audio
# def text_to_audio(client, text, audio_path):
#     response=client.audio.speech.create(model="tts-1", voice="alloy", input=text)
#     response.stream_to_file(audio_path)  

# convert text to audio
def text_to_audio(client, text, audio_path):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(audio_path)
    
                              
# def main():
#     # # Set the title of the app
#     # st.title("AI Voice Assistant")
    
#     # # Create a subheader
#     # st.subheader("Talk to your AI assistant and get instant responses.")
    
#     # # Initialize the audio recorder
#     # audio_data = audio_recorder()
    
#     # # Display the audio recorder widget
#     # st.audio(audio_data, format="audio/wav")
    
#     # # If audio data is available, process it
#     # if audio_data:
#     #     # Convert audio data to text using OpenAI's Whisper API
#     #     openai.api_key = "YOUR_OPENAI_API_KEY"
#     #     response = openai.Audio.transcribe(
#     #         model="whisper-1",
#     #         file=audio_data,
#     #         response_format="text"
#     #     )   
    
#     st.sidebar.title("API KEY CONFIGURATION")
#     api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

#     st.title("🎤💬 AI Voice Assistant")
#     st.write("Hi! I'm your AI voice assistant. How can I help you today?")
#     st.write("Click on the Voice Recorder to interact with me.")

#     #check if the API key is provided
#     if api_key:
#         # Initialize the OpenAI client
#         openai_client = setup_openai_client(api_key)

#         # Initialize the Gemini client
#         gemini_client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)

#         # Display the audio recorder widget
#         st.write("Click the button below to start recording your voice.")
#         audio_data = audio_recorder()

#         # If audio data is available, process it
#         if api_key:
#             client = setup_openai_client(api_key)
#             recorded_audio = audio_recorder()

# if __name__ == "__main__":
#     main()  

def main():
    st.sidebar.title("API KEY CONFIGURATION")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    st.title("🎤💬 AI Voice Assistant")
    st.write("Hi! I'm your AI voice assistant. How can I help you today?")
    st.write("Click on the Voice Recorder to interact with me.")

    #check if the API key is provided
    if api_key:
        # Initialize the OpenAI client
        openai_client = setup_openai_client(api_key)

        # Initialize the Gemini client
        gemini_client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)

        # Display the audio recorder widget
        st.write("Click the button below to start recording your voice.")
        audio_data = audio_recorder()

        # If audio data is available, process it
        if audio_data:
            client = setup_openai_client(api_key)
            # Process the recorded audio here using 'audio_data'
            st.audio(audio_data, format="audio/wav") # Example of displaying the recorded audio

            audio_file = "audio.wav"

            with open(audio_file, "wb") as f:
                f.write(audio_data)

            transcribed_text= transcribe_audio_with_openai(client, audio_file)
            
            ai_response = fetch_ai_response(client, transcribed_text)
            response_audio_file = "audio_response.wav"
            text_to_audio(client, ai_response, response_audio_file)
            st.audio(response_audio_file, format="audio/wav")
            st.write("AI Response:", ai_response)

st.markdown("---")
st.markdown("© Copyright 2025, created by Jose Ambrosio")

if __name__ == "__main__":
    main()