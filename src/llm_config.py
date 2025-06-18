from langchain_openai import ChatOpenAI
import os
import streamlit as st


openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY as a Hugging Face Space secret.")
    print(f"DEBUG llm_config: OpenAI API key NOT loaded") # DEBUG print

print(f"DEBUG llm_config: OpenAI API key loaded status: {bool(openai_api_key)}") # DEBUG print

# --- OpenAI LLMs ---
# Use st.cache_resource to avoid re-initializing these models on every Streamlit rerun
@st.cache_resource
def get_openai_llm(model_name: str, temperature: float = 0, max_tokens: int = None):
    """Initializes and caches an OpenAI ChatModel."""
    if not openai_api_key:
        print(f"DEBUG llm_config: Skipping OpenAI model initialization for {model_name}: API key is missing (in func).")
        return None
    print(f"Initializing OpenAI model: {model_name}")
    return ChatOpenAI(
        temperature=temperature,
        model=model_name,
        openai_api_key=openai_api_key,
        max_tokens=max_tokens
    )

print("DEBUG llm_config: Initializing base LLM...")
llm = get_openai_llm(model_name="gpt-4o-mini", temperature=0, max_tokens=2000)

print("DEBUG llm_config: Initializing memory LLM...")
llm_memory = get_openai_llm(model_name="gpt-4o-mini", max_tokens=2000)

print("DEBUG llm_config: Initializing agent LLM...")
llm_agent = get_openai_llm(model_name="gpt-4o-mini", temperature=0, max_tokens=2000)
