import streamlit as st
from google.generativeai import GenerativeModel, configure
import requests
from bs4 import BeautifulSoup
import textwrap
import os
from dotenv import load_dotenv
import time
import random
from urllib.parse import urlparse
import httpx
import asyncio

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(page_title="Multi-URL Content Analyzer", page_icon="🔍", layout="wide")
st.title("🔍 AI Multi-URL Content Analyzer")
st.markdown("Extract and analyze content from multiple webpages")

# Initialize Gemini
def init_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in .env file")
        st.stop()
    configure(api_key=api_key)
    return GenerativeModel('gemini-2.0-flash-exp-image-generation')

model = init_gemini()

# Session state to persist content
if 'extracted_contents' not in st.session_state:
    st.session_state.extracted_contents = []
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'auto_qa' not in st.session_state:
    st.session_state.auto_qa = []

async def extract_content(url):
    methods = [
        _scrape_with_httpx,
        _scrape_with_requests,
    ]
    
    for method in methods:
        try:
            content = await method(url)
            if content: return content
        except Exception as e:
            continue
    
    return None

async def _scrape_with_httpx(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/',
        'DNT': '1'
    }
    
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return await _process_html(response.text)

async def _scrape_with_requests(url):
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Mozilla/5.0 (X11; Linux x86_64)'
        ]),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    time.sleep(random.uniform(1, 3))
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return await _process_html(response.text)

async def _process_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    for tag in ['script', 'style', 'nav', 'footer', 'iframe', 'noscript', 'svg']:
        for element in soup.find_all(tag):
            element.decompose()
    
    article = soup.find('article') or soup.find('main') or soup.find('div', class_=lambda x: x and 'content' in x.lower())
    if article:
        text = '\n\n'.join(p.get_text().strip() for p in article.find_all(['p', 'h1', 'h2', 'h3']) if p.get_text().strip())
        return text if text else soup.get_text()
    return soup.get_text()

def generate_response(prompt, contents):
    try:
        combined_content = "\n\n".join([f"Content from URL {i+1}:\n{content[:10000]}" for i, content in enumerate(contents)])
        
        full_prompt = f"""
        Combined content from multiple URLs:
        {combined_content}
        
        {prompt}
        
        Please provide a detailed answer based on all the content. 
        If the information is not available in the content, state that clearly.
        Reference which URL the information came from when possible.
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_auto_qa(contents):
    combined_content = "\n\n".join([f"Content from URL {i+1}:\n{content[:10000]}" for i, content in enumerate(contents)])
    
    prompt = f"""
    Combined content from multiple URLs:
    {combined_content}
    
    Generate 5 important questions and answers that would help someone understand this combined content.
    Format as:
    Q1: [question]
    A1: [answer] (Source: URL #)
    
    Q2: [question]
    A2: [answer] (Source: URL #)
    
    ...
    """
    
    response = model.generate_content(prompt)
    return response.text

def main():
    st.sidebar.header("URL Management")
    
    # URL input with dynamic addition
    url_inputs = []
    num_urls = st.sidebar.number_input("Number of URLs to analyze", min_value=1, max_value=5, value=1)
    
    for i in range(num_urls):
        url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}", placeholder="https://example.com")
        if url:
            url_inputs.append(url)
    
    if st.sidebar.button("Extract Content"):
        if not url_inputs:
            st.warning("Please enter at least one URL")
            return
            
        st.session_state.extracted_contents = []
        with st.spinner(f"Extracting content from {len(url_inputs)} URLs..."):
            for url in url_inputs:
                content = asyncio.run(extract_content(url))
                if content:
                    st.session_state.extracted_contents.append(content)
            
            if not st.session_state.extracted_contents:
                st.error("Failed to extract content from any URLs. Try different URLs or methods.")
                return
                
            st.session_state.qa_history = []
            
            # Generate automatic Q&A
            with st.spinner("Generating automatic Q&A..."):
                st.session_state.auto_qa = generate_auto_qa(st.session_state.extracted_contents)
    
    # Display extracted contents if they exist
    if st.session_state.extracted_contents:
        st.header("Extracted Contents")
        
        # Show tabs for each URL's content
        tabs = st.tabs([f"URL {i+1}" for i in range(len(st.session_state.extracted_contents))])
        for i, content in enumerate(st.session_state.extracted_contents):
            with tabs[i]:
                st.text_area(f"Content from URL {i+1}", 
                           value=content[:5000] + ("..." if len(content) > 5000 else ""), 
                           height=300,
                           key=f"content_{i}")
        
        # Analysis options
        st.header("Analysis Options")
        analysis_type = st.radio("Choose analysis type:", 
                               ["Summary", "Key Points", "Pre-generated Q&A", "Interactive Q&A Chat"], 
                               horizontal=False,
                               key="analysis_type")
        
        if analysis_type == "Summary":
            if st.button("Generate Summary"):
                with st.spinner("Generating combined summary..."):
                    response = generate_response("Provide a comprehensive summary combining all content", st.session_state.extracted_contents)
                    st.subheader("Combined Summary Results")
                    st.markdown(response)
        
        elif analysis_type == "Key Points":
            if st.button("Generate Key Points"):
                with st.spinner("Generating combined key points..."):
                    response = generate_response("Provide bullet points of the key information across all content", st.session_state.extracted_contents)
                    st.subheader("Combined Key Points Results")
                    st.markdown(response)
        
        elif analysis_type == "Pre-generated Q&A":
            st.subheader("Automatically Generated Q&A")
            st.markdown(st.session_state.auto_qa)
            
            if st.button("Regenerate Q&A"):
                with st.spinner("Generating new Q&A set..."):
                    st.session_state.auto_qa = generate_auto_qa(st.session_state.extracted_contents)
                    st.rerun()
        
        elif analysis_type == "Interactive Q&A Chat":
            st.markdown("### Ask Questions About the Combined Content")
            user_question = st.text_input("Enter your question:", key="user_question")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Get Answer"):
                    if not user_question:
                        st.warning("Please enter a question")
                    else:
                        with st.spinner("Generating answer..."):
                            answer = generate_response(user_question, st.session_state.extracted_contents)
                            st.session_state.qa_history.append((user_question, answer))
            with col2:
                if st.button("Clear History"):
                    st.session_state.qa_history = []
                    st.rerun()
            
            # Display Q&A history
            if st.session_state.qa_history:
                st.markdown("### Conversation History")
                for i, (question, answer) in enumerate(st.session_state.qa_history, 1):
                    st.markdown(f"**You:** {question}")
                    st.markdown(f"**AI:** {answer}")
                    st.divider()

st.markdown("---")
st.markdown("© Copyright 2025, created by Jose Ambrosio")

if __name__ == "__main__":
    main()


