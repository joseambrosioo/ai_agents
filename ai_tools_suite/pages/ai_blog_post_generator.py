import os
import streamlit as st
# from google import genai
import google.generativeai as genai
from google.generativeai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from streamlit_carousel import carousel
import tempfile

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize Gemini client
# client = genai.Client(api_key=GOOGLE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')

single_image = dict(
    title="",
    text="",
    interval=None,
    img=""
)

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
    'response_modalities': ['TEXT', 'IMAGE'] # Include the modalities here
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

# Set app to wide mode
# st.set_page_config(layout="wide")

# Title of the app
st.set_page_config(page_title="AI Blog Post Generator", page_icon="✍")
st.title('✍📝 AI Blog Post Generator')

# Create a subheader
st.subheader('Write amazing blog posts with the help of AI.')

# Sidebar for user input
with st.sidebar:
    # Header for sidebar
    st.title('Blog Post Details')
    st.subheader('Enter the following details to generate your blog post:')

    blog_title = st.text_input('Blog Post Title')
    blog_keywords = st.text_area('Blog Post Keywords (comma-separated)')
    num_words = st.slider('Number of Words', min_value=200, max_value=500, step=100)
    num_images = st.number_input('Number of Images', min_value=1, max_value=5, step=1)

    prompt_parts = [f"Generate a comprehensive, engaging blog post relevant to the given title \"{blog_title}\" and keywords \"{blog_keywords}\". Make sure to incorporate these keywords in the blog post. The blog should be approximately {num_words} words in length, suitable for an online audience. Ensure the content is original, informative, and maintains a consistent tone throughout."]

    # Submit button to generate blog post
    submit_button = st.button('Generate Blog Post')

# If the submit button is clicked, generate the blog post
if submit_button:
    # response_text = client.models.generate_content(
    response_text = model.generate_content(
        # model="gemini-2.0-flash-exp-image-generation",
        contents=prompt_parts,
        # config=generation_config,
        generation_config=generation_config,
        # safety_settings=safety_settings
    ).text

    # st.subheader("Generated Blog Post:")

    image_gallery = []

    # Generate content using Gemini API for images
    for i in range(num_images):
        image_prompt = (f'Generate a 3D Rendered Image of: {blog_title}')
        # image_response = client.models.generate_content(
        image_response = model.generate_content(
            # model="gemini-2.0-flash-exp-image-generation",
            contents=image_prompt,
            generation_config=generation_config
            # config=types.GenerateContentConfig(
                # response_modalities=['TEXT', 'IMAGE'] # Request only image
            # )
        )

        # Extract and append the image data to the gallery
        for part in image_response.candidates[0].content.parts:
            if part.inline_data is not None:
                new_image = single_image.copy()
                # new_image['title'] = f"Image {i+1}"
                # new_image['text'] = f"{blog_title}"
                try:
                    image_bytes = part.inline_data.data
                    image = Image.open(BytesIO(image_bytes))
                    # Save the image to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        image.save(tmp_file.name)
                        new_image['img'] = tmp_file.name
                    image_gallery.append(new_image)
                except Exception as e:
                    st.error(f"Error processing image {i+1}: {e}")

    if image_gallery:
        carousel(items=image_gallery, width=1) # Adjust height as needed
        # Clean up temporary files (optional, but good practice)
        for item in image_gallery:
            if os.path.exists(item['img']):
                os.remove(item['img'])
    else:
        st.info("No images were generated.")
        
    st.write(response_text)

st.markdown("---")
st.markdown("© Copyright 2025, created by Jose Ambrosio")