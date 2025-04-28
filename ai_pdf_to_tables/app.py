import streamlit as st
from google.generativeai import GenerativeModel, configure
import os
from dotenv import load_dotenv
import PyPDF2
import tempfile
import pandas as pd
from io import StringIO

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(page_title="PDF to Structured Data Converter", page_icon="📊")
st.title("📊 PDF to Structured Data Converter")
st.markdown("Extract tables and structured records from PDF documents using AI")

# Initialize Gemini
def init_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in .env file")
        st.stop()
    configure(api_key=api_key)
    return GenerativeModel('gemini-2.0-flash-exp-image-generation')

model = init_gemini()

# Session state to persist data
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []
if 'processed_tables' not in st.session_state:
    st.session_state.processed_tables = {}

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

def extract_structured_data(text, extraction_type="tables"):
    try:
        prompt = f"""
        The following text was extracted from a PDF document. Please analyze it and extract structured data.
        
        Extraction type: {extraction_type}
        
        For tables:
        - Identify all tables in the text
        - Extract each table with headers and rows
        - Format as CSV with proper escaping
        - Label each table with a descriptive title
        
        For records:
        - Identify structured records (like products, employees, transactions, etc.)
        - Extract each record with consistent fields
        - Format as CSV with proper escaping
        - Include a header row with field names
        
        Text content:
        {text[:15000]}  # Limiting to first 15k chars to avoid token limits
        
        Important:
        - Only return the extracted data in CSV format
        - Include a header line before each table/record set
        - For tables, add a comment line above each table like: # Table 1: [description]
        - For records, add a comment line like: # Records: [type of records]
        - If no structured data is found, return "# No structured data found"
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def parse_extracted_data(raw_data):
    tables = {}
    current_table = None
    current_data = []
    
    # Split the raw data into lines
    lines = raw_data.split('\n')
    
    for line in lines:
        if line.startswith('# Table') or line.startswith('# Records'):
            # Save previous table if exists
            if current_table and current_data:
                try:
                    # Convert collected lines to a CSV string
                    csv_str = '\n'.join(current_data)
                    # Read into pandas dataframe
                    df = pd.read_csv(StringIO(csv_str))
                    tables[current_table] = df
                except Exception as e:
                    tables[current_table] = f"Error parsing table: {str(e)}"
            
            # Start new table
            current_table = line[2:].strip()  # Remove the '# '
            current_data = []
        elif line.strip() and not line.startswith('#'):
            current_data.append(line)
    
    # Add the last table if exists
    if current_table and current_data:
        try:
            csv_str = '\n'.join(current_data)
            df = pd.read_csv(StringIO(csv_str))
            tables[current_table] = df
        except Exception as e:
            tables[current_table] = f"Error parsing table: {str(e)}"
    
    return tables

def main():
    st.sidebar.header("PDF Upload")
    
    # PDF upload section
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF document", 
        type=["pdf"],
        help="Upload a PDF containing tables or structured data"
    )
    
    extraction_type = st.sidebar.radio(
        "Extraction Type",
        ["Auto Detect", "Tables", "Records"],
        index=0
    )
    
    if st.sidebar.button("Extract Structured Data"):
        if not uploaded_file:
            st.warning("Please upload a PDF file")
            return
            
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("Failed to extract text from PDF")
                return
            
            st.session_state.extracted_data = []
            
            # Determine extraction type
            actual_type = "tables" if extraction_type == "Auto Detect" else extraction_type.lower()
            
            with st.spinner(f"Extracting {actual_type}..."):
                raw_data = extract_structured_data(text, actual_type)
                if "# No structured data found" in raw_data:
                    st.warning("No structured data found in the document")
                    return
                
                st.session_state.processed_tables = parse_extracted_data(raw_data)
    
    # Display extracted data if available
    if st.session_state.processed_tables:
        st.header("Extracted Structured Data")
        
        # Show tabs for each table
        tabs = st.tabs(list(st.session_state.processed_tables.keys()))
        
        for tab, (table_name, table_data) in zip(tabs, st.session_state.processed_tables.items()):
            with tab:
                st.subheader(table_name)
                
                if isinstance(table_data, pd.DataFrame):
                    # Display dataframe
                    st.dataframe(table_data)
                    
                    # Add download button
                    csv = table_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download {table_name} as CSV",
                        data=csv,
                        file_name=f"{table_name.replace(':', '').replace(' ', '_')}.csv",
                        mime='text/csv',
                        key=f"dl_{table_name}"
                    )
                else:
                    st.error(table_data)  # Display error message
        
        # Show raw extracted data
        with st.expander("View Raw Extraction Output"):
            raw_output = "\n".join([
                f"{name}\n{df.to_csv(index=False) if isinstance(df, pd.DataFrame) else df}\n\n" 
                for name, df in st.session_state.processed_tables.items()
            ])
            st.text_area("Raw Output", raw_output, height=300)

st.markdown("---")
st.markdown("© Copyright 2025, created by Jose Ambrosio")


if __name__ == "__main__":
    main()