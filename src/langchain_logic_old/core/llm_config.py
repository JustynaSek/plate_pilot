from langchain_openai import ChatOpenAI
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from huggingface_hub import login
import streamlit as st # Import streamlit for caching


# --- API Key Configuration ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY as a Hugging Face Space secret.")
  
# --- OpenAI LLMs ---
# Use st.cache_resource to avoid re-initializing these models on every Streamlit rerun
@st.cache_resource
def get_openai_llm(model_name: str, temperature: float = 0, max_tokens: int = None):
    """Initializes and caches an OpenAI ChatModel."""
    if not openai_api_key:
        return None
    print(f"Initializing OpenAI model: {model_name}")
    return ChatOpenAI(
        temperature=temperature,
        model=model_name,
        openai_api_key=openai_api_key,
        max_tokens=max_tokens
    )

llm = get_openai_llm(model_name="gpt-4o-mini", temperature=0, max_tokens=500)
llm_memory = get_openai_llm(model_name="gpt-4o-mini") # temperature defaults to 0.7 if not set
llm_agent = get_openai_llm(model_name="ggpt-4o-mini", temperature=0)

FINE_TUNED_MODEL_ID = "ft:gpt-4o-mini-2024-07-18:justyna-sek:plate-pilot:BaG94pm0"
llm_fine_tuned_gpt4o = get_openai_llm(model_name=FINE_TUNED_MODEL_ID, temperature=0)


# --- Qwen Model Configuration and Loading ---
QWEN_FINE_TUNED_MODEL_ID = "JustynaSek86/health-advisor-qwen-1-8b-chat-ft-merged"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        # add_to_git_credential=False is good for web apps on Spaces
        login(token=hf_token, add_to_git_credential=False)
        print("Logged in to Hugging Face Hub for Qwen model loading.")
    except Exception as e:
        print(f"Warning: Could not log in to Hugging Face Hub for Qwen. Error: {e}")
else:
    print("HF_TOKEN environment variable not found. Qwen model loading might fail if private or rate-limited.")

# BitsAndBytes configuration for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

llm_fine_tuned_qwen = None
qwen_tokenizer = None

# Use st.cache_resource for the Qwen model and tokenizer
@st.cache_resource
def load_qwen_model_and_tokenizer():
    """Loads and caches the fine-tuned Qwen model and its tokenizer."""
    try:
        print(f"Loading Qwen tokenizer from: {QWEN_FINE_TUNED_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(QWEN_FINE_TUNED_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Consistent with training

        print(f"Loading merged Qwen fine-tuned model from: {QWEN_FINE_TUNED_MODEL_ID}...")
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_FINE_TUNED_MODEL_ID,
            quantization_config=bnb_config, # Apply quantization during model load
            torch_dtype=torch.bfloat16, # Use bfloat16 for computation
            device_map="auto", # Automatically map to available devices (GPU/CPU)
            trust_remote_code=True # Required for Qwen models
        )
        model.eval() # Set model to evaluation mode

        print("Creating HuggingFace pipeline for Qwen model...")
        qwen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # device=0 if torch.cuda.is_available() else -1, # device_map="auto" usually handles this, but can specify for fine-grained control
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        llm_pipeline = HuggingFacePipeline(pipeline=qwen_pipeline)
        print("Fine-tuned Qwen model and Langchain pipeline loaded successfully.")
        return llm_pipeline, tokenizer

    except Exception as e:
        print(f"FATAL ERROR: Could not load the fine-tuned Qwen model. Check model ID, HF_TOKEN, internet, and GPU setup. Error: {e}")
        return None, None # Ensure None if loading fails

llm_fine_tuned_qwen, qwen_tokenizer = load_qwen_model_and_tokenizer()
