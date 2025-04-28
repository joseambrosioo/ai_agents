import streamlit as st
from google.generativeai import GenerativeModel, configure
import os
from dotenv import load_dotenv
import PyPDF2
import io
import textwrap
import tempfile

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(page_title="AI Multi-PDF Analyzer", page_icon="📄")
st.title("📄🔎 AI Multi-PDF Analyzer")
st.markdown("Upload and analyze multiple PDF documents with AI.")

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
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'auto_qa' not in st.session_state:
    st.session_state.auto_qa = []

def extract_text_from_pdf(pdf_file):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        
        # Read the PDF
        with open(tmp_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
        
        # Clean up
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def generate_response(prompt, texts):
    try:
        combined_content = "\n\n".join([f"Content from Document {i+1}:\n{text[:10000]}" for i, text in enumerate(texts)])
        
        full_prompt = f"""
        Combined content from multiple PDF documents:
        {combined_content}
        
        {prompt}
        
        Please provide a detailed answer based on all the documents. 
        If the information is not available in the documents, state that clearly.
        Reference which document the information came from when possible.
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_auto_qa(texts):
    combined_content = "\n\n".join([f"Content from Document {i+1}:\n{text[:10000]}" for i, text in enumerate(texts)])
    
    prompt = f"""
    Combined content from multiple PDF documents:
    {combined_content}
    
    Generate 5 important questions and answers that would help someone understand these documents.
    Format as:
    Q1: [question]
    A1: [answer] (Source: Document #)
    
    Q2: [question]
    A2: [answer] (Source: Document #)
    
    ...
    """
    
    response = model.generate_content(prompt)
    return response.text

def main():
    st.sidebar.header("PDF Management")
    
    # PDF upload section
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload multiple PDFs for combined analysis"
    )
    
    if st.sidebar.button("Extract Text"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file")
            return
            
        st.session_state.extracted_texts = []
        with st.spinner(f"Extracting text from {len(uploaded_files)} PDFs..."):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if text:
                    st.session_state.extracted_texts.append(text)
            
            if not st.session_state.extracted_texts:
                st.error("Failed to extract text from any PDFs. Try different files.")
                return
                
            st.session_state.qa_history = []
            
            # Generate automatic Q&A
            with st.spinner("Generating automatic Q&A..."):
                st.session_state.auto_qa = generate_auto_qa(st.session_state.extracted_texts)
    
    # Display extracted texts if they exist
    if st.session_state.extracted_texts:
        st.header("Extracted Text Contents")
        
        # Show tabs for each document's content
        tabs = st.tabs([f"Document {i+1}" for i in range(len(st.session_state.extracted_texts))])
        for i, text in enumerate(st.session_state.extracted_texts):
            with tabs[i]:
                st.text_area(f"Text from Document {i+1}", 
                           value=text[:5000] + ("..." if len(text) > 5000 else ""), 
                           height=300,
                           key=f"text_{i}", disabled=True)
        
        # Analysis options
        st.header("Analysis Options")
        analysis_type = st.radio("Choose analysis type:", 
                               ["Summary", "Key Points", "Pre-generated Q&A", "Interactive Q&A Chat"], 
                               horizontal=False,
                               key="analysis_type")
        
        if analysis_type == "Summary":
            if st.button("Generate Summary"):
                with st.spinner("Generating combined summary..."):
                    response = generate_response("Provide a comprehensive summary combining all documents", st.session_state.extracted_texts)
                    st.subheader("Combined Summary Results")
                    st.markdown(response)
        
        elif analysis_type == "Key Points":
            if st.button("Generate Key Points"):
                with st.spinner("Generating combined key points..."):
                    response = generate_response("Provide bullet points of the key information across all documents", st.session_state.extracted_texts)
                    st.subheader("Combined Key Points Results")
                    st.markdown(response)
        
        elif analysis_type == "Pre-generated Q&A":
            st.subheader("Automatically Generated Q&A")
            st.markdown(st.session_state.auto_qa)
            
            if st.button("Regenerate Q&A"):
                with st.spinner("Generating new Q&A set..."):
                    st.session_state.auto_qa = generate_auto_qa(st.session_state.extracted_texts)
                    st.rerun()
        
        elif analysis_type == "Interactive Q&A Chat":
            st.markdown("### Ask Questions About the Documents")
            user_question = st.text_input("Enter your question:", key="user_question")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Get Answer"):
                    if not user_question:
                        st.warning("Please enter a question")
                    else:
                        with st.spinner("Generating answer..."):
                            answer = generate_response(user_question, st.session_state.extracted_texts)
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