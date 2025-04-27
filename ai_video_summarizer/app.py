import os
import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# Load environment variables from .env file (optional, but good practice)
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Function to extract video ID from YouTube URL
def get_video_id(youtube_url):
    """Extracts the video ID from a YouTube URL."""
    if "youtube.com/watch?v=" in youtube_url:
        return youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]
    return None

# Function to get transcript from video ID
def get_youtube_transcript(video_id):
    """Fetches the transcript for a given YouTube video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([d['text'] for d in transcript_list])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Function to summarize text using Gemini 1.5 Flash
def summarize_text_with_gemini(text, max_length=500):
    """Summarizes the provided text using the Gemini 1.5 Flash model."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = f"Summarize the following text, focusing on the main points and key information. Keep the summary concise, ideally under {max_length} words:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error summarizing text with Gemini: {e}")
        return None

# Streamlit App Title
st.title("📝AI Youtube Video Summarizer")

# Input field for YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL:", "")

# Button to trigger summarization
if st.button("Summarize"):
    if youtube_url:
        video_id = get_video_id(youtube_url)
        if video_id:
            st.info("Fetching transcript...")
            transcript = get_youtube_transcript(video_id)

            if transcript:
                st.info("Generating summary...")
                summary = summarize_text_with_gemini(transcript)

                if summary:
                    st.subheader("Summary:")
                    st.write(summary)
                else:
                    st.warning("Could not generate summary.")
            else:
                st.warning("Could not retrieve transcript for this video. It might not have captions.")
        else:
            st.warning("Invalid YouTube URL. Please enter a valid URL.")
    else:
        st.warning("Please enter a YouTube URL.")

st.markdown("---")
st.markdown("© Copyright 2025, created by Jose Ambrosio")
