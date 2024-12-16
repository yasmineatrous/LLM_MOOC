import streamlit as st
import os
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import tempfile
from pinecone import Pinecone
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Set up environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize embeddings and Pinecone
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Pinecone index
index_name = "rag"
index = pc.Index(index_name)
# Print index statistics
index.describe_index_stats()

# Function to process and upload the PDF
def process_and_upload_pdf(uploaded_file):
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load the PDF from the temporary file path
    loader = PyPDFLoader(temp_file_path)
    data = loader.load()

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    fragments = text_splitter.split_documents(data)

    # Convert fragments into embeddings and store them in Pinecone
    pinecone_store = PineconeVectorStore.from_documents(
        fragments, embeddings, index_name=index_name
    )

    # Optionally, remove the temporary file after processing
    os.remove(temp_file_path)

# Streamlit UI
st.set_page_config(
    page_title="Company Documents Upload",
    page_icon="📂",
    layout="wide",
)

# Title and description
st.title("📂 Upload and Process Company Documents")
st.markdown(
    "Upload company documents to process and store them for seamless integration with the RAG system."
)

# File upload functionality
uploaded_pdf = st.file_uploader(" ", type=["pdf"])

if uploaded_pdf is not None:
    st.write("Processing PDF...")
    process_and_upload_pdf(uploaded_pdf)
    st.success("PDF successfully processed and rules uploaded to Pinecone!")
else:
    st.write(" ")
