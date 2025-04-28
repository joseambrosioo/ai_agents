import streamlit as st
from google.generativeai import GenerativeModel, configure
import os
from dotenv import load_dotenv
import PyPDF2
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import csv

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(page_title="AI Multi-File Analyzer", page_icon="📊")
st.title("📊🔎 AI Multi-File Analyzer")
st.markdown("Upload and analyze multiple PDF and CSV documents with AI.")

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
if 'uploaded_dfs' not in st.session_state:
    st.session_state.uploaded_dfs = []
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'auto_qa' not in st.session_state:
    st.session_state.auto_qa = []

def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        
        with open(tmp_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
        
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def process_csv(file):
    try:
        # Try reading with pandas first
        try:
            df = pd.read_csv(file)
            return df
        except:
            # Fallback to manual CSV reading if pandas fails
            file.seek(0)
            decoded_file = file.read().decode('utf-8')
            reader = csv.reader(decoded_file.splitlines())
            data = list(reader)
            df = pd.DataFrame(data[1:], columns=data[0])
            return df
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

def generate_response(prompt, texts=None, dfs=None):
    try:
        content_parts = []
        
        if texts:
            combined_text = "\n\n".join([f"Content from Document {i+1}:\n{text[:10000]}" for i, text in enumerate(texts)])
            content_parts.append(f"Text Content from PDF Documents:\n{combined_text}")
        
        if dfs:
            df_descriptions = []
            for i, df in enumerate(dfs):
                df_desc = f"""
                Data from CSV Document {i+1}:
                - Shape: {df.shape}
                - Columns: {', '.join(df.columns)}
                - First 5 rows:
                {df.head().to_string()}
                """
                df_descriptions.append(df_desc)
            content_parts.append(f"Data from CSV Documents:\n{'\n\n'.join(df_descriptions)}")
        
        full_content = "\n\n".join(content_parts)
        
        full_prompt = f"""
        {full_content}
        
        {prompt}
        
        Please provide a detailed answer based on all the documents. 
        For CSV data, include specific insights from the data when possible.
        If the information is not available in the documents, state that clearly.
        Reference which document the information came from when possible.
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_auto_qa(texts=None, dfs=None):
    content_parts = []
    
    if texts:
        combined_text = "\n\n".join([f"Content from Document {i+1}:\n{text[:10000]}" for i, text in enumerate(texts)])
        content_parts.append(f"Text Content from PDF Documents:\n{combined_text}")
    
    if dfs:
        df_descriptions = []
        for i, df in enumerate(dfs):
            df_desc = f"""
            Data from CSV Document {i+1}:
            - Shape: {df.shape}
            - Columns: {', '.join(df.columns)}
            - First 5 rows:
            {df.head().to_string()}
            """
            df_descriptions.append(df_desc)
        content_parts.append(f"Data from CSV Documents:\n{'\n\n'.join(df_descriptions)}")
    
    full_content = "\n\n".join(content_parts)
    
    prompt = f"""
    {full_content}
    
    Generate 5 important questions and answers that would help someone understand these documents.
    For CSV data, include questions about trends, patterns, and insights from the data.
    Format as:
    Q1: [question]
    A1: [answer] (Source: Document #)
    
    Q2: [question]
    A2: [answer] (Source: Document #)
    
    ...
    """
    
    response = model.generate_content(prompt)
    return response.text

def generate_data_analysis(dfs):
    analysis_results = []
    
    for i, df in enumerate(dfs):
        analysis = f"## Analysis for CSV Document {i+1}\n"
        
        # Basic stats
        analysis += "### Basic Statistics\n"
        analysis += df.describe().to_string() + "\n\n"
        
        # Generate visualizations
        analysis += "### Visualizations\n"
        
        # For each numeric column, create a histogram
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            try:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                plt.title(f'Distribution of {col}')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                st.image(buf, caption=f'Distribution of {col}')
                plt.close()
            except Exception as e:
                analysis += f"\nCould not generate histogram for {col}: {str(e)}\n"
        
        # Generate correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)
                plt.title('Correlation Heatmap')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                st.image(buf, caption='Correlation Heatmap')
                plt.close()
            except Exception as e:
                analysis += f"\nCould not generate correlation heatmap: {str(e)}\n"
        
        # Ask Gemini for insights
        prompt = f"""
        Here is the data from CSV Document {i+1}:
        - Columns: {', '.join(df.columns)}
        - First 5 rows:
        {df.head().to_string()}
        
        Please provide:
        1. Key insights from this data
        2. Interesting patterns or trends
        3. Potential data quality issues
        4. Suggestions for further analysis
        """
        
        try:
            gemini_response = model.generate_content(prompt)
            analysis += "\n### AI-Generated Insights\n"
            analysis += gemini_response.text
        except Exception as e:
            analysis += f"\nCould not generate AI insights: {str(e)}\n"
        
        analysis_results.append(analysis)
    
    return analysis_results

def perform_eda(df):
    st.subheader("Exploratory Data Analysis")
    
    # Basic info
    st.write("### Dataset Overview")
    st.write(f"**Shape:** {df.shape}")
    
    # Display the dataframe
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Column selector
    st.write("### Column Information")
    selected_col = st.selectbox("Select a column to analyze:", df.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Column statistics
        st.write("**Column Statistics**")
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            st.write(df[selected_col].describe())
        else:
            st.write(df[selected_col].describe(include='object'))
    
    with col2:
        # Unique values
        st.write("**Unique Values**")
        unique_vals = df[selected_col].unique()
        st.write(f"Count: {len(unique_vals)}")
        if len(unique_vals) <= 20:
            st.write(unique_vals)
    
    # Visualization section
    st.write("### Visualizations")
    viz_type = st.selectbox("Select visualization type:", 
                           ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot", "Correlation Heatmap"])
    
    if viz_type == "Histogram" and pd.api.types.is_numeric_dtype(df[selected_col]):
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)
    
    elif viz_type == "Box Plot" and pd.api.types.is_numeric_dtype(df[selected_col]):
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_col], ax=ax)
        st.pyplot(fig)
    
    elif viz_type == "Bar Chart":
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            fig, ax = plt.subplots()
            df[selected_col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            df[selected_col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
    
    elif viz_type == "Scatter Plot" and len(df.select_dtypes(include=['number']).columns) >= 2:
        x_col = st.selectbox("Select X-axis column:", df.select_dtypes(include=['number']).columns)
        y_col = st.selectbox("Select Y-axis column:", df.select_dtypes(include=['number']).columns)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        st.pyplot(fig)
    
    elif viz_type == "Correlation Heatmap" and len(df.select_dtypes(include=['number']).columns) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, ax=ax)
        st.pyplot(fig)
    
    # Missing values
    st.write("### Missing Values Analysis")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.write("**Columns with missing values:**")
        st.write(missing[missing > 0])
        
        # Visualize missing values
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, ax=ax)
        st.pyplot(fig)
    else:
        st.success("No missing values found in the dataset!")

def main():
    st.sidebar.header("File Management")
    
    # File upload section
    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload multiple PDFs for combined analysis"
    )
    
    uploaded_csvs = st.sidebar.file_uploader(
        "Upload CSV documents", 
        type=["csv"], 
        accept_multiple_files=True,
        help="Upload multiple CSVs for data analysis"
    )
    
    if st.sidebar.button("Process Files"):
        st.session_state.extracted_texts = []
        st.session_state.uploaded_dfs = []
        
        if uploaded_pdfs:
            with st.spinner(f"Extracting text from {len(uploaded_pdfs)} PDFs..."):
                for file in uploaded_pdfs:
                    text = extract_text_from_pdf(file)
                    if text:
                        st.session_state.extracted_texts.append(text)
        
        if uploaded_csvs:
            with st.spinner(f"Processing {len(uploaded_csvs)} CSVs..."):
                for file in uploaded_csvs:
                    df = process_csv(file)
                    if df is not None:
                        st.session_state.uploaded_dfs.append(df)
        
        if not st.session_state.extracted_texts and not st.session_state.uploaded_dfs:
            st.error("Failed to process any files. Try different files.")
            return
            
        st.session_state.qa_history = []
        
        # Generate automatic Q&A
        with st.spinner("Generating automatic Q&A..."):
            st.session_state.auto_qa = generate_auto_qa(
                st.session_state.extracted_texts if st.session_state.extracted_texts else None,
                st.session_state.uploaded_dfs if st.session_state.uploaded_dfs else None
            )
    
    # Display extracted content if it exists
    if st.session_state.extracted_texts or st.session_state.uploaded_dfs:
        st.header("Uploaded Content")
        
        # Show PDF content
        if st.session_state.extracted_texts:
            st.subheader("PDF Documents")
            pdf_tabs = st.tabs([f"PDF {i+1}" for i in range(len(st.session_state.extracted_texts))])
            for i, text in enumerate(st.session_state.extracted_texts):
                with pdf_tabs[i]:
                    st.text_area(f"Text from PDF {i+1}", 
                               value=text[:5000] + ("..." if len(text) > 5000 else ""), 
                               height=300,
                               key=f"pdf_text_{i}", disabled=True)
        
        # Show CSV content
        if st.session_state.uploaded_dfs:
            st.subheader("CSV Documents")
            csv_tabs = st.tabs([f"CSV {i+1}" for i in range(len(st.session_state.uploaded_dfs))])
            for i, df in enumerate(st.session_state.uploaded_dfs):
                with csv_tabs[i]:
                    st.write(f"Shape: {df.shape}")
                    st.dataframe(df.head())
        
        # Analysis options
        st.header("Analysis Options")
        analysis_type = st.radio("Choose analysis type:", 
                               ["Summary", "Key Points", "Pre-generated Q&A", 
                                "Interactive Q&A Chat", "Data Analysis (CSV only)", "EDA (CSV only)"], 
                               horizontal=False,
                               key="analysis_type")
        
        if analysis_type == "Summary":
            if st.button("Generate Summary"):
                with st.spinner("Generating combined summary..."):
                    response = generate_response(
                        "Provide a comprehensive summary combining all documents",
                        st.session_state.extracted_texts if st.session_state.extracted_texts else None,
                        st.session_state.uploaded_dfs if st.session_state.uploaded_dfs else None
                    )
                    st.subheader("Combined Summary Results")
                    st.markdown(response)
        
        elif analysis_type == "Key Points":
            if st.button("Generate Key Points"):
                with st.spinner("Generating combined key points..."):
                    response = generate_response(
                        "Provide bullet points of the key information across all documents",
                        st.session_state.extracted_texts if st.session_state.extracted_texts else None,
                        st.session_state.uploaded_dfs if st.session_state.uploaded_dfs else None
                    )
                    st.subheader("Combined Key Points Results")
                    st.markdown(response)
        
        elif analysis_type == "Pre-generated Q&A":
            st.subheader("Automatically Generated Q&A")
            st.markdown(st.session_state.auto_qa)
            
            if st.button("Regenerate Q&A"):
                with st.spinner("Generating new Q&A set..."):
                    st.session_state.auto_qa = generate_auto_qa(
                        st.session_state.extracted_texts if st.session_state.extracted_texts else None,
                        st.session_state.uploaded_dfs if st.session_state.uploaded_dfs else None
                    )
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
                            answer = generate_response(
                                user_question,
                                st.session_state.extracted_texts if st.session_state.extracted_texts else None,
                                st.session_state.uploaded_dfs if st.session_state.uploaded_dfs else None
                            )
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
        
        elif analysis_type == "Data Analysis (CSV only)":
            if not st.session_state.uploaded_dfs:
                st.warning("No CSV files uploaded for analysis")
            else:
                st.subheader("Data Analysis Results")
                with st.spinner("Analyzing CSV data..."):
                    analysis_results = generate_data_analysis(st.session_state.uploaded_dfs)
                    for result in analysis_results:
                        st.markdown(result)
        
        elif analysis_type == "EDA (CSV only)":
            if not st.session_state.uploaded_dfs:
                st.warning("No CSV files uploaded for EDA")
            else:
                selected_df_index = st.selectbox(
                    "Select CSV to analyze:",
                    options=range(len(st.session_state.uploaded_dfs)),
                    format_func=lambda x: f"CSV {x+1}"
                )
                selected_df = st.session_state.uploaded_dfs[selected_df_index]
                perform_eda(selected_df)

if __name__ == "__main__":
    main()