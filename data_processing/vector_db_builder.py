import json
import os
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
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

# Hardcoded paths
processed_recipes_file = 'data_processing/data/recipes_minimal.json'
vector_db_dir = 'data_processing/vector_db'

documents = load_and_process_recipes(processed_recipes_file)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

try:
    print(f"Using vector DB directory: {vector_db_dir}")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=vector_db_dir)
except Exception as e:
    print(f"Error creating vectorstore: {e}")
    vectorstore = None

print("Vectorstore created successfully.", vectorstore)
