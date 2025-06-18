---
title: PlatePilot - Your AI Meal Plan Assistant
emoji: 🍽️
colorFrom: blue
colorTo: green
sdk: docker
app_file: src/test_app.py 
python_version: "3.12"
---

# PlatePilot - AI-Powered Meal Planning Assistant

PlatePilot is an intelligent meal planning assistant that helps you create personalized meal plans based on your dietary preferences, restrictions, and health goals. The app uses advanced AI to generate customized meal plans, suggest dessert recipes, and provide health tips.

## Features

- **Personalized Meal Plans**: Get customized meal plans based on your dietary restrictions and preferences
- **Smart Modifications**: Easily modify your meal plan with specific requests
- **Dessert Suggestions**: Get dessert recipes that match your dietary preferences
- **Health Tips**: Receive personalized health and nutrition advice
- **Dietary Restrictions Support**: Works with various dietary needs (general, diabetes, heart failure, low-sodium, vegetarian)

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (required for AI functionality)
- gdown (for downloading the vector database)

## Installation

### Using pip (Traditional Method)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PlatePilot.git //todo update
cd PlatePilot
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the vector database:
```bash
# Install gdown if not already installed
pip install gdown

# Download the vector database
gdown https://drive.google.com/uc?id=1E0D8l_RSFrv37QZvam9fiYpA8HQoVRIU

# Extract the database
# On Windows
tar -xf vector_db.tar.gz -C data_processing/

# On macOS/Linux
tar -xzf vector_db.tar.gz -C data_processing/
```

### Using uv (Faster Alternative)

1. Install uv if you haven't already:
```bash
pip install uv
```

2. Clone the repository:
```bash
git clone https://github.com/JustynaSek/plate_pilot.git
cd PlatePilot
```

3. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

4. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

5. Download the vector database:
```bash
# Install gdown if not already installed
pip install gdown

# Download the vector database
gdown https://drive.google.com/uc?id=1E0D8l_RSFrv37QZvam9fiYpA8HQoVRIU

# Extract the database
# On Windows
tar -xf vector_db.tar.gz -C data_processing/

# On macOS/Linux
tar -xzf vector_db.tar.gz -C data_processing/
```

## Configuration

1. Set up your OpenAI API key:
```bash
# On Windows
set OPENAI_API_KEY=your-api-key-here


# On macOS/Linux
export OPENAI_API_KEY=your-api-key-here
```

## Running the App

1. Make sure your virtual environment is activated and you have set your OpenAI API key.

2. Start the Streamlit app:
```bash
streamlit run src/streamlit_app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Using the App

1. **Enter Your Preferences**:
   - Your name (optional)
   - Dietary restrictions
   - Food preferences
   - Foods to avoid

2. **Generate a Meal Plan**:
   - Click "Get My Meal Plan" to generate your first meal plan
   - The plan will be displayed with detailed recipes and instructions

3. **Modify Your Plan**:
   - If you want to make changes, enter your modification request
   - Click "Update Meal Plan" to get a modified version
   - Both original and updated plans will be displayed

4. **Get Dessert Recipes**:
   - Click "Get Dessert Recipes" to receive dessert suggestions
   - Recipes will match your dietary preferences

5. **Get Health Tips**:
   - Click "Get Health Tips" for personalized nutrition advice
   - Tips are tailored to your dietary restrictions

## Troubleshooting

- **API Key Issues**: Ensure your OpenAI API key is correctly set in the environment variables
- **Dependency Conflicts**: If you encounter dependency conflicts, try using uv instead of pip
- **Rate Limiting**: If you see rate limit errors, wait a few minutes before trying again

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.