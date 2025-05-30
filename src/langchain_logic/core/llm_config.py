from langchain_openai import ChatOpenAI
import os
import torch # Required for tensor types and GPU checks
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from huggingface_hub import login 

openai_api_key = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    max_tokens=500
)

llm_memory = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo"
)

llm_agent = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

FINE_TUNED_MODEL_ID = "ft:gpt-4o-mini-2024-07-18:justyna-sek:plate-pilot:BaG94pm0"
llm_fine_tuned_gpt4o = ChatOpenAI(
    temperature=0,
    model_name=FINE_TUNED_MODEL_ID,
    openai_api_key=openai_api_key
)

QWEN_FINE_TUNED_MODEL_ID = "JustynaSek86/health-advisor-qwen-1-8b-chat-ft-merged"
BASE_QWEN_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False) # add_to_git_credential=False is good for web apps
        print("Logged in to Hugging Face Hub for Qwen model loading.")
    except Exception as e:
        print(f"Warning: Could not log in to Hugging Face Hub for Qwen. Error: {e}")
else:
    print("HF_TOKEN environment variable not found for Qwen model loading.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

llm_fine_tuned_qwen = None
qwen_tokenizer = None

try:
    print(f"Loading Qwen tokenizer from: {QWEN_FINE_TUNED_MODEL_ID}...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_FINE_TUNED_MODEL_ID, trust_remote_code=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    qwen_tokenizer.padding_side = "right" # Consistent with training

    print(f"Loading merged Qwen fine-tuned model from: {QWEN_FINE_TUNED_MODEL_ID}...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_FINE_TUNED_MODEL_ID,
        quantization_config=bnb_config, # Apply quantization during model load
        torch_dtype=torch.bfloat16, # Use bfloat16 for computation
        device_map="auto", # Automatically map to available devices (GPU/CPU)
        trust_remote_code=True # Required for Qwen models
    )
    qwen_model.eval() # Set model to evaluation mode

    print("Creating HuggingFace pipeline for Qwen model...")
    qwen_pipeline = pipeline(
        "text-generation",
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        device=0 if torch.cuda.is_available() else -1, # Use GPU 0 if available, else CPU
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        eos_token_id=qwen_tokenizer.eos_token_id,
        pad_token_id=qwen_tokenizer.pad_token_id
    )

    llm_fine_tuned_qwen = HuggingFacePipeline(pipeline=qwen_pipeline)
    print("Fine-tuned Qwen model and Langchain pipeline loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR: Could not load the fine-tuned Qwen model. Check model ID, HF_TOKEN, internet, and GPU setup. Error: {e}")
    llm_fine_tuned_qwen = None # Ensure it's None if loading fails
