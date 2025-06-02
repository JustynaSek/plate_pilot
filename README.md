---
title: PlatePilot - Your AI Meal Plan Assistant
emoji: 🍽️
colorFrom: blue
colorTo: green
sdk: docker
app_file: src/test_app.py 
python_version: "3.12"
---

# PlatePilot - Your AI Meal Plan Assistant

This is an AI-powered meal planning application that provides personalized meal plans and health tips. It leverages Retrieval Augmented Generation (RAG) with a ChromaDB vector store, an agent with web search capabilities, and multiple Large Language Models including OpenAI's GPT-4o (standard and fine-tuned) and a fine-tuned Qwen model.

**Features:**
* **Personalized Meal Plans:** Get meal plans tailored to your dietary restrictions, likes, and dislikes.
* **Meal Plan Modification:** Request changes to generated meal plans.
* **Dessert Recipe Search:** Find dessert recipes using a web search agent.
* **Health Tips:** Receive health advice from both a fine-tuned GPT-4o model and a fine-tuned Qwen model.

**Technologies Used:**
* Streamlit (for the interactive UI)
* LangChain (for orchestrating LLM interactions, RAG, and agents)
* OpenAI GPT-4o (LLM)
* Qwen-1.8B-Chat (fine-tuned LLM)
* ChromaDB (Vector Store for RAG)
* Hugging Face Transformers (for Qwen model loading)
* SearchAPI (for web search tool)

**How to Use:**
1.  Enter your name (optional).
2.  Select your dietary restrictions and specify food likes/dislikes.
3.  Click "Get My Meal Plan" to receive a personalized meal plan.
4.  You can then request modifications or search for dessert recipes.
5.  Get general or restriction-specific health tips.