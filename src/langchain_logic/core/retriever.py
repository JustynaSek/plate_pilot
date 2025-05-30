import os
import configparser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

config = configparser.ConfigParser()
config.read('config/config.ini')

db_persist_dir = 'data_processing/' + config['database']['persist_directory']
vectordb = Chroma(persist_directory=db_persist_dir, embedding_function=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY")))
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, 'lambda_mult':0.3})
