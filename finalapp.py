import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from pathlib import Path
from pdf2image import convert_from_path

# Directly setting the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = "..."

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./IP_PDF")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Set custom CSS styles for background colors, responsive buttons, and input field dimensions
custom_css = """
<style>
    .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    .css-1d391kg {  /* Sidebar style */
        background-color: #76B900;
    }
    .button-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 20px;
    }
    .button-container button {
        padding: 12px 24px;  /* Add padding to buttons */
        font-size: 16px;
        border-radius: 5px;
        border: none;
        background-color: #76B900;
        color: white;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .button-container button:hover {
        background-color: #5a9b00;
    }
    .button-container button:active {
        transform: scale(0.98);
    }
    @media (max-width: 768px) {
        .button-container button {
            width: 100%;
            font-size: 14px;
            padding: 10px 20px;  /* Adjust padding for smaller screens */
        }
    }
    .text-input input {
        width: 100%;
        height: 150px;  /* Increased height */
        font-size: 18px;  /* Increased font size */
        padding: 15px;  /* Increased padding */
        box-sizing: border-box;
        border-radius: 5px;
        border: 1px solid #76B900;
        color: #000000;
        background-color: #FFFFFF;
    }
    .status-message {
        margin-left: 20px;
        font-size: 16px;
        color: #76B900;
    }
</style>
"""

# Inject custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Display the PDFs in the sidebar with custom background color
pdf_dir = Path("./IP_PDF")
pdf_files = [f for f in pdf_dir.glob("*.pdf")]

# Create a folder to store the thumbnails
thumbnail_dir = Path("./thumbnails")
thumbnail_dir.mkdir(exist_ok=True)

# Display each PDF in the sidebar with a thumbnail and download link
st.sidebar.title("Available PDFs")

for pdf_file in pdf_files:
    # Generate a thumbnail if it doesn't exist
    thumbnail_path = thumbnail_dir / f"{pdf_file.stem}.png"
    if not thumbnail_path.exists():
        images = convert_from_path(pdf_file, first_page=1, last_page=1)
        images[0].save(thumbnail_path, 'PNG')

    # Display the thumbnail and download link
    with st.sidebar:
        st.image(str(thumbnail_path), width=100)
        st.markdown(f"**{pdf_file.name}**")
        with open(pdf_file, "rb") as pdf:
            pdf_data = pdf.read()
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=pdf_file.name,
                mime="application/pdf"
            )
        st.write("------------------")

# Main content
st.title("Intellectual Property Chatbot Nvidia-NIM")

# Create a row for text input and buttons
col1, col2 = st.columns([3, 1])  # Create two columns: one for text input and one for the button

# Apply custom styling to text input field
with col1:
    prompt1 = st.text_input(
        "Enter Your Question From Documents",
        max_chars=500,
        key="prompt1",
        help="Type your question here.",
        label_visibility="visible",
        placeholder="Type your question here..."
    )

# Place the "Submit" button below the text input
submit_button = st.button("Submit", key="submit_button", help="Click to submit your question")

# Display the "Vector Store DB Is Ready" message next to the "Documents Embedding" button
col3, col4 = st.columns([3, 1])  # Create columns for buttons and status message

with col3:
    documents_embedding_button = st.button("Documents Embedding", key="documents_embedding", help="Click to embed documents")

with col4:
    status_message = st.empty()  # Create an empty container for the status message

# Update status message when the "Documents Embedding" button is clicked
if documents_embedding_button:
    vector_embedding()
    status_message.markdown("<p class='status-message'>Vector Store DB Is Ready</p>", unsafe_allow_html=True)

# Handle the submit button click
if submit_button:
    if prompt1:
        llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start}")
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
