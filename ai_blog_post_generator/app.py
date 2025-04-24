import os
import streamlit as st
from google import genai
from google.genai import types
# import google.generativeai as genai
# from openai import OpenAI
# from apikey import GOOGLE_GEMINI_API_KEY, OPENAI_API_KEY
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY")

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)

# genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)

generation_config = {
    "temperature": 0.9, 
    "top_p": 1, 
    "top_k": 1, 
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# model = genai.GenerativeModel(model_name="gemini-2.5-pro-preview-03-25",
# model = genai.GenerativeModel(model_name="gemini-2.5-pro-exp-03-25",
# model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp",
# model = client.GenerativeModel(model_name="gemini-2.0-flash-exp",
#                             generation_config=generation_config,
#                             safety_settings=safety_settings)


#Set app to wide mode
st.set_page_config(layout="wide")

#title of the app
st.title('✍📝 AI Blog Post Generator')

#create a subheader
st.subheader('Write amazing blog posts with the help of AI.')

#sidebar for user input
with st.sidebar:
    # Header for sidebar
    st.title('Blog Post Details')
    st.subheader('Enter the following details to generate your blog post:')
    
    # Text input for blog title
    # blog_title = st.text_input('Blog Title', 'Enter your blog title here')  
    blog_title = st.text_input('Blog Post Title')  

    # Keyword input for blog title
    # blog_keywords = st.text_area('Blog Keywords (comma-separated)', 'Enter your blog keywords here (comma-separated)')
    blog_keywords = st.text_area('Blog Post Keywords (comma-separated)')

    # Number of words input for blog post
    num_words = st.slider('Number of Words', min_value=200, max_value=500, step=100)

    # Number of images input for blog post
    num_images = st.number_input('Number of Images', min_value=1, max_value=5, step=1)

    prompt_parts = [f"Generate a comprehensive, engaging blog post relevant to the given title \"{blog_title}\" and keywords \"{blog_keywords}\". Make sure to incorporate these keywords in the blog post. The blog should be approximately {num_words} words in length, suitable for an online audience. Ensure the content is original, informative, and maintains a consistent tone throughout."]

    # response = model.generate_content(prompt_parts)

    # Submit button to generate blog post
    submit_button = st.button('Generate Blog Post')


# If the submit button is clicked, generate the blog post   
if submit_button:
    # st.image("https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwzNjUyOXwwfDF8c2VhcmNofDJ8fGJsb2clMjBwb3N0ZXJ8ZW58MHx8fHwxNjg5NTY1NzA3&ixlib=rb-4.0.3&q=80&w=1080")

    # response = model.generate_content(prompt_parts)

    response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=prompt_parts,
    config=generation_config,
    # safety_settings=safety_settings
    )

	# Define the request contents for image generation
    image_prompt = (f'Generate a 3D Rendered Image of: {blog_title}')

    # Generate content using Gemini API
    image_response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=image_prompt,
        config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE']
        # response_modalities=['IMAGE']
        )
    )

    # Extract and display the image using Streamlit
    for part in image_response.candidates[0].content.parts:
        if part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            st.image(image)


    # st.title("YOUR BLOG POST")
    
    st.write(response.text)