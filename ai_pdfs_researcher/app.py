import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter  
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

    return text

def get_text_chunks(text):
    """Splits text into chunks of specified size with overlap."""
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len, 
    )
    chunks = text_splitter.split_text(text) 
    return chunks


def get_vectorstore(text_chunks): # Function to create vectorstore
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")  # Use HuggingFaceInstructEmbeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  # Create a FAISS vectorstore from text chunk
    return vectorstore


def get_conversation_chain(vectorstore):  # Function to create conversation chain
    llm = ChatOpenAI()  
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.1, "max_lenght": 512})  # Use HuggingFaceHub for LLM

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Initialize memory for conversation
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),  # Use the vectorstore as a retriever
        memory=memory,  # Use the memory for conversation history
    )
    return conversation_chain
  

def handle_user_input(user_question):
    response = st.session_state.conversation("question", user_question)  # Get response from conversation chain
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]  # Update chat history in session state

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()  # Load environment variables from .env file
    st.set_page_config(page_title="AI PDFs Researcher", page_icon="📄🔎", layout="wide")

    st.write(css, unsafe_allow_html=True)  # Load custom CSS for styling
    # Add your app logic here
    # For example, you can add file upload, text input, etc.

    if "conversation" not in st.session_state:
        st.session_state.conversartion = None
    if "chat_history" not in st.session_state:
        st.session_state.conversartion = None

    st.header("📄🔎 AI PDFs Researcher")
    st.text_input("Ask a question about your documents:")

    st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)  # Load user template for chat interface
    st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)  # Load user template for chat interface
   
    with st.sidebar:
        st.subheader("Your documents:")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process PDFs"):
            st.spinner("Processing...")
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)  # Function to extract text from PDFs
            st.write(raw_text)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)  # Function to split text into chunks
            st.write(text_chunks)
            # create vectorstore using OpenAIEmbeddings  
            vectorstore = get_vectorstore(text_chunks)  # Function to create vectorstore
              
            # create conversation chain using OpenAIEmbeddings
            st.session_state.conversation = get_conversation_chain(vectorstore)


        # st.title("Upload Your PDF Files")
        # uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        # if uploaded_files:
        #     for uploaded_file in uploaded_files:
        #         st.write(f"Uploaded file: {uploaded_file.name}")

if __name__ == "__main__":
    main()