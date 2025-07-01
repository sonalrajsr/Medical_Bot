from Utility.get_data_from_pinecone import get_result_pinecone
from Utility.pdf_chunk import split_text
from Utility.pdf_loader import load_pdf
from Utility.pinecone_vector_db import store_vector_db
from Utility.LLM import get_llm_response
import streamlit as st

st.set_page_config(page_title="Medical Chat Bot", page_icon=":robot_face:", layout="wide")
def main():
    st.title("Medical Chat Bot")
    st.write("Upload a PDF document to create a knowledge base and ask questions.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            pdf_text = load_pdf(uploaded_file)
            chunks = split_text(pdf_text)
            store_vector_db(chunks)
            st.success("PDF processed and indexed successfully!")

    query = st.text_input("Ask a question about the medical document:")
    
    if query:
        with st.spinner("Searching for relevant information..."):
            result = get_result_pinecone(query)
            LLM_response = get_llm_response(query, result)
            st.write(LLM_response)

if __name__ == "__main__":
    main()