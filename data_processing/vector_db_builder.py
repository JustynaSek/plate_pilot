import json
import os
import configparser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def load_recipes(filepath="recipes.json"):
    """Loads the recipes from the JSON file."""
    try:
        with open(filepath, "r") as f:
            recipes = json.load(f)
            return recipes
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please ensure the file exists.")
        return []  # Return an empty list to avoid further errors

def process_recipe(recipe):
    """Formats a recipe dictionary into a single text string."""
    text = f"Title: {recipe['title']}\n\nDescription: {recipe['description']}\n\nIngredients:\n"
    text += "\n".join(f"- {ingredient}" for ingredient in recipe['ingredients']) + "\n\nInstructions:\n"
    text += "\n".join(f"{i+1}. {instruction}" for i, instruction in enumerate(recipe['instructions'])) + "\n\nSuitable For: " + ", ".join(recipe['suitable_for']) + "\n\nLikes: " + ", ".join(recipe['likes']) + "\n\nDislikes: " + ", ".join(recipe['dislikes'])
    return text

def load_and_process_recipes(filepath="recipes.json"):
    """Loads, processes, and prepares recipes for a vector database."""
    recipes = load_recipes(filepath)
    documents = []
    for recipe in recipes:
        text = process_recipe(recipe)
        metadata = {
            "title": recipe['title'],
            "suitable_for": ", ".join(recipe['suitable_for']),
            "likes": ", ".join(recipe['likes']),
            "dislikes": ", ".join(recipe['dislikes']),
        }
        # print(f"Processing recipe, text: {text}")
        # print(f"Processing recipe,Metadata: {metadata}")
        documents.append(Document(page_content=text, metadata=metadata))
        # print(f"documens: {documents}")
    return documents

config = configparser.ConfigParser()
config.read('config/config.ini')

documents = load_and_process_recipes('data_processing/data/'+config['database']['processed_recipes_file'])

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))  # Replace with your chosen embedding model

try:
    db_persist_dir = config['database']['persist_directory']
    print(f"Persist directory from config: {db_persist_dir}")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory='data_processing/'+db_persist_dir)
except KeyError as e:
    print(f"Error: Section or key not found in config file: {e}")
    vectorstore = None

print("Vectorstore created successfully.", vectorstore)
