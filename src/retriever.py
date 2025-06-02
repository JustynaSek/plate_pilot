# src/retriever.py
import os
import configparser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st # Ensure this is imported for caching

config = configparser.ConfigParser()
# Use absolute path relative to Dockerfile WORKDIR /app, as you've set up
config.read('/app/config/config.ini') 

db_persist_dir = '/app/data_processing/' + config['database']['persist_directory'] 

# Use st.cache_resource to ensure these are only loaded/initialized once per session
@st.cache_resource
def get_retriever_and_vectordb():
    print("DEBUG retriever: Starting OpenAIEmbeddings and ChromaDB initialization...") # DEBUG print
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("DEBUG retriever: OpenAI API key NOT found for embeddings. This will cause an error if not resolved.") # DEBUG print
        st.error("OpenAI API key not found for embeddings. Please set OPENAI_API_KEY secret.")
        raise ValueError("OpenAI API key is missing for embeddings.") # This will stop the app if key is missing

    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    print(f"DEBUG retriever: OpenAIEmbeddings initialized. Loading ChromaDB from: {db_persist_dir}...") # DEBUG print
    
    # Allow dangerous deserialization if your ChromaDB was created with a different LangChain version
    vectordb = Chroma(persist_directory=db_persist_dir, embedding_function=embeddings_model, allow_dangerous_deserialization=True) 
    
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, 'lambda_mult':0.3})
    print("DEBUG retriever: ChromaDB and Retriever initialized successfully.") # DEBUG print
    return retriever, vectordb # Return both if needed later, or just retriever

retriever, vectordb_instance = get_retriever_and_vectordb() # Call the cached function
print("DEBUG retriever: Retriever and Vectordb instance assigned.") # DEBUG print