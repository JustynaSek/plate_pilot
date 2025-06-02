from .prompts.health_tips_chat_prompt import HEALTH_TIPS_PROMPT, QWEN_SYSTEM_MESSAGE
from llm_config import qwen_tokenizer as global_qwen_tokenizer

class HealthAdvisor:

    def __init__(self, llm_gpt4o, llm_qwen):
        self.llm_gpt4o = llm_gpt4o
        self.llm_qwen = llm_qwen
        self.qwen_tokenizer = global_qwen_tokenizer


    def _construct_base_health_query(self, dietary_restrictions: str) -> str:
        """
        Constructs the core health tips query based on dietary restrictions.
        """
        if dietary_restrictions != "general":
            return f"one tip for a person with {dietary_restrictions} health condition."
        else:
            return f"one general health tip."


    def generate_health_tips_gpt4o(self, user_name, dietary_restrictions):
        print(' dietary_restrictions: ' + dietary_restrictions) # Added space for readability
        health_tips_chain = (
            {
                "user_full_query": lambda x: (
                    f"Give me one tip for person with {x['dietary_restrictions']} health condition."
                    if x["dietary_restrictions"] != "general"
                    else f"Give me general health tips."
                ),
                "user_name": lambda x: x["user_name"],
            }
            | HEALTH_TIPS_PROMPT
            | self.llm_gpt4o # Use self.llm_gpt4o here
        )
        
        user_input = {"user_name": user_name, "dietary_restrictions": dietary_restrictions}
        response = health_tips_chain.invoke(user_input)
        print("Health tips response (GPT-4o):", response.content) # Added model indicator
        return response.content
    
    def generate_health_tips_qwen(self, user_name: str, dietary_restrictions: str) -> str:
        """
        Generates health tips using the fine-tuned Qwen model.
        """
        print(f"Generating health tips using Qwen for {user_name} with dietary_restrictions: {dietary_restrictions}")

        if not self.llm_qwen:
            print("Qwen model (llm_qwen) not loaded. Cannot generate tips.")
            return "Error: Qwen model is not available. Please try the GPT-4o option."
        if not self.qwen_tokenizer: # Safety check for tokenizer as well
            print("Qwen tokenizer not loaded. Cannot generate tips.")
            return "Error: Qwen tokenizer is not available. Please try the GPT-4o option."


        # Construct the user message content that matches Qwen's fine-tuning format
        user_message_content_qwen = self._construct_base_health_query(dietary_restrictions)

        # Build the conversation list for Qwen's chat template
        conversation_qwen = [
            {"role": "system", "content": QWEN_SYSTEM_MESSAGE}, 
            {"role": "user", "content": user_message_content_qwen},
        ]

        # Apply Qwen's specific chat template to get the final prompt string
        formatted_qwen_prompt = self.qwen_tokenizer.apply_chat_template(
            conversation_qwen,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            raw_qwen_response = self.llm_qwen.invoke(formatted_qwen_prompt)

            # Manually parse the Qwen response to extract only the assistant's part
            assistant_prefix = f"<|im_start|>assistant\n"
            if assistant_prefix in raw_qwen_response:
                start_index = raw_qwen_response.rfind(assistant_prefix) + len(assistant_prefix)
                final_qwen_response = raw_qwen_response[start_index:].strip()
                if final_qwen_response.endswith("<|im_end|>"):
                    final_qwen_response = final_qwen_response[:-len("<|im_end|>")].strip()
                if final_qwen_response.endswith(self.qwen_tokenizer.eos_token):
                    final_qwen_response = final_qwen_response[:-len(self.qwen_tokenizer.eos_token)].strip()
            else:
                final_qwen_response = raw_qwen_response.strip() # Fallback

            # Add personalization: dynamically add "Hello {user_name}!" to the response
            if user_name and user_name.lower() != "user":
                final_qwen_response = f"Hello {user_name}! " + final_qwen_response
            
            print("Health tips response (Qwen):", final_qwen_response)
            return final_qwen_response
        except Exception as e:
            print(f"Error generating Qwen health tips: {e}")
            return "Error: Could not generate health tips from Qwen. Please try again."