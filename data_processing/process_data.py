import json
from nltk.stem import WordNetLemmatizer
import nltk

def remove_newlines(text):
    if isinstance(text, str):
        return text.replace('\n', ' ')
    elif isinstance(text, list):
        return [remove_newlines(item) for item in text]
    return text

def download_nltk_resources():
    nltk.download('punkt_tab')
    try:
        nltk.data.find('tokenizers/punkt/PY3/english.pickle')
        print("NLTK Punkt tokenizer data found.")
    except LookupError:
        print("NLTK Punkt tokenizer data not found. Downloading...")
        nltk.download('punkt')
        print("NLTK Punkt tokenizer data downloaded successfully.")

    try:
        nltk.data.find('corpora/stopwords')
        print("NLTK Stopwords corpus found.")
    except LookupError:
        print("NLTK Stopwords corpus not found. Downloading...")
        nltk.download('stopwords')
        print("NLTK Stopwords corpus downloaded successfully.")

    try:
        nltk.data.find('corpora/wordnet')
        print("NLTK WordNet corpus found.")
    except LookupError:
        print("NLTK WordNet corpus not found. Downloading...")
        nltk.download('wordnet')
        print("NLTK WordNet corpus downloaded successfully.")

def lemmatize_text(text):
    words = nltk.word_tokenize(text.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()] # Keep only alphanumeric
    return " ".join(lemmatized_words)

def process_data(recipe):
    recipe['title'] = remove_newlines(recipe['title'])
    recipe['description'] = remove_newlines(recipe['description'])
    recipe['ingredients'] = [remove_newlines(ing) for ing in recipe['ingredients']]
    recipe['instructions'] = [remove_newlines(inst) for inst in recipe['instructions']]

    recipe['description'] = lemmatize_text(recipe['description'])
    recipe['ingredients'] = [lemmatize_text(ing) for ing in recipe['ingredients']]
    recipe['instructions'] = [lemmatize_text(inst) for inst in recipe['instructions']]
    
    return recipe

def load_and_process_recipes_minimal(filepath):
    with open(filepath, "r") as f:
        recipes = json.load(f)
    processed_recipes = [process_data(recipe) for recipe in recipes]
    return processed_recipes

# Hardcoded paths
recipes_file = "data_processing/data/recipes.json"
minimal_recipes_file = "data_processing/data/recipes_minimal.json"

download_nltk_resources()
lemmatizer = WordNetLemmatizer()

minimal_recipes = load_and_process_recipes_minimal(recipes_file)

# Save the minimal version without newlines
with open(minimal_recipes_file, "w") as f:
    json.dump(minimal_recipes, f, indent=2)

print(json.dumps(minimal_recipes[0], indent=2))