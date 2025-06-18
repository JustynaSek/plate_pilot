import os
import subprocess
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st 

print("DEBUG retriever: Module retriever.py is being loaded.")

def download_vector_db():
    """Download the vector database from Google Drive if it doesn't exist."""
    data_processing_base_path = os.path.join(find_project_root(), 'data_processing')
    vector_db_path = os.path.join(data_processing_base_path, 'vector_db')
    
    if not os.path.exists(vector_db_path):
        print("Vector database not found. Downloading...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            
            download_url = "https://drive.google.com/uc?id=1E0D8l_RSFrv37QZvam9fiYpA8HQoVRIU"
            subprocess.check_call(["gdown", download_url])
            
            os.makedirs(data_processing_base_path, exist_ok=True)
            
            if os.name == 'nt':  # Windows
                subprocess.check_call(["tar", "-xf", "vector_db.tar.gz", "-C", data_processing_base_path])
            else:  # Unix-like
                subprocess.check_call(["tar", "-xzf", "vector_db.tar.gz", "-C", data_processing_base_path])
            
            os.remove("vector_db.tar.gz")
            print("Vector database downloaded and extracted successfully.")
        except Exception as e:
            print(f"Error downloading vector database: {str(e)}")
            st.error("Failed to download vector database. Please follow the manual download instructions in the README.")
            raise

def find_project_root():
    """
    Dynamically finds the project root directory.
    Assumes 'src' is directly under the project root.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_dir) # project_root/ (e.g., PlatePilot/)
    return project_root

@st.cache_resource
def get_global_retriever():
    """Initializes and caches the OpenAIEmbeddings and ChromaDB retriever."""
    print("DEBUG retriever: Starting OpenAIEmbeddings and ChromaDB initialization inside get_global_retriever().")
    
    if os.path.exists('/app/data_processing'):
        data_processing_base_path = '/app/data_processing'
        print("DEBUG retriever: Running in deployed (Docker) environment.")
    else:
        root = find_project_root()
        data_processing_base_path = os.path.join(root, 'data_processing')
        print(f"DEBUG retriever: Running in local environment. Project root: {root}")
        
        download_vector_db()
    
    db_persist_dir = os.path.join(data_processing_base_path, 'vector_db')
    print(f"DEBUG retriever: DB persist directory: {db_persist_dir}")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("DEBUG retriever: OpenAI API key NOT found for embeddings. This will cause an error if not resolved.")
        st.error("OpenAI API key not found for embeddings. Please set OPENAI_API_KEY secret.")
        raise ValueError("OpenAI API key is missing for embeddings.")

    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    print(f"DEBUG retriever: OpenAIEmbeddings initialized. Loading ChromaDB from: {db_persist_dir}...")
    
    # Allow dangerous deserialization if your ChromaDB was created with a different LangChain version
    vectordb = Chroma(persist_directory=db_persist_dir, embedding_function=embeddings_model) 
    
    retriever_instance = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, 'lambda_mult':0.3})
    print("DEBUG retriever: ChromaDB and Retriever initialized successfully.")
    return retriever_instance