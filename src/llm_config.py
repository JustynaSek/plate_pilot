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

from huggingface_hub import login, HfHubEndpointError, HfHubHTTPError, Repository # Added HfHub specific errors

# Use st.cache_resource for the Qwen model and tokenizer
@st.cache_resource
def load_qwen_model_and_tokenizer():
    """Loads and caches the fine-tuned Qwen model and its tokenizer."""
    # --- TEMPORARILY DISABLED QWEN LOADING FOR TESTING ---
    print("--- Qwen model loading is temporarily disabled for testing. ---")
    return None, None 
    # # Pre-checks for model ID and token
    # if not QWEN_FINE_TUNED_MODEL_ID:
    #     print("FATAL ERROR: QWEN_FINE_TUNED_MODEL_ID is not set.")
    #     return None, None
    
    # # Add a check for HF_TOKEN if the model is private or if auth is generally problematic
    # # This might be redundant with the login() call but adds clarity
    # if not hf_token and not Repository(QWEN_FINE_TUNED_MODEL_ID).is_public:
    #     print(f"FATAL ERROR: Model {QWEN_FINE_TUNED_MODEL_ID} might be private, but HF_TOKEN is not available.")
    #     return None, None

    # tokenizer = None
    # model = None
    # llm_pipeline = None

    # try:
    #     print(f"Attempting to load Qwen tokenizer from: {QWEN_FINE_TUNED_MODEL_ID}...")
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         QWEN_FINE_TUNED_MODEL_ID, 
    #         trust_remote_code=True,
    #         token=hf_token # Pass token explicitly to tokenizer
    #     )
    #     if tokenizer.pad_token is None:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.padding_side = "right"

    #     print(f"Attempting to load Qwen model from: {QWEN_FINE_TUNED_MODEL_ID}...")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         QWEN_FINE_TUNED_MODEL_ID,
    #         quantization_config=bnb_config,
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #         trust_remote_code=True,
    #         token=hf_token # Pass token explicitly to model
    #     )
    #     model.eval()

    #     print("Creating HuggingFace pipeline for Qwen model...")
    #     qwen_pipeline = pipeline(
    #         "text-generation",
    #         model=model,
    #         tokenizer=tokenizer,
    #         max_new_tokens=256,
    #         do_sample=True,
    #         temperature=0.7,
    #         top_p=0.9,
    #         top_k=50,
    #         eos_token_id=tokenizer.eos_token_id,
    #         pad_token_id=tokenizer.pad_token_id
    #     )

    #     llm_pipeline = HuggingFacePipeline(pipeline=qwen_pipeline)
    #     print("Fine-tuned Qwen model and Langchain pipeline loaded successfully.")
    #     return llm_pipeline, tokenizer

    # # Catch specific exceptions for better debugging
    # except (HfHubEndpointError, HfHubHTTPError) as e:
    #     print(f"FATAL ERROR: Hugging Face Hub connectivity/authentication error when loading Qwen model/tokenizer: {e}")
    #     if "401 Client Error" in str(e) or "403 Client Error" in str(e):
    #         print("This usually means your HF_TOKEN is invalid or lacks access to the model.")
    #     elif "404 Client Error" in str(e):
    #         print("This usually means the model ID is incorrect or the model doesn't exist.")
    #     return None, None
    # except ImportError as e:
    #     print(f"FATAL ERROR: Missing dependency for Qwen model loading. Ensure accelerate, bitsandbytes, and torch are installed. Error: {e}")
    #     return None, None
    # except ValueError as e:
    #     print(f"FATAL ERROR: Invalid value or configuration during Qwen model loading: {e}")
    #     if "device_map" in str(e) or "bitsandbytes" in str(e):
    #         print("Check GPU availability/configuration or bitsandbytes setup.")
    #     return None, None
    # except TypeError as e: # Keep catching the generic TypeError as a fallback
    #     print(f"FATAL ERROR: Unexpected TypeError during Qwen model loading. This might indicate a deeper environment issue or data corruption. Error: {e}")
    #     return None, None
    # except Exception as e: # Generic catch-all for any other unexpected errors
    #     print(f"FATAL ERROR: An unexpected error occurred while loading Qwen model: {e}")
    #     return None, None
llm_fine_tuned_qwen, qwen_tokenizer = load_qwen_model_and_tokenizer()
