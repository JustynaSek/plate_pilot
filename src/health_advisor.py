from prompts.health_tips_chat_prompt import HEALTH_TIPS_PROMPT

class HealthAdvisor:

    def __init__(self, llm_gpt4o):
        self.llm_gpt4o = llm_gpt4o

    def _construct_base_health_query(self, dietary_restrictions: str) -> str:
        """
        Constructs the core health tips query based on dietary restrictions.
        """
        if dietary_restrictions != "general":
            return f"one tip for a person with {dietary_restrictions} health condition."
        else:
            return f"one general health tip."

    def generate_health_tips_gpt4o(self, user_name, dietary_restrictions):
        print(' dietary_restrictions: ' + dietary_restrictions)
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
            | self.llm_gpt4o
        )
        
        user_input = {"user_name": user_name, "dietary_restrictions": dietary_restrictions}
        response = health_tips_chain.invoke(user_input)
        print("Health tips response (GPT-4o):", response.content)
        return response.content