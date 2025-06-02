from .prompts.meal_plan_chat_prompt import MEAL_PLAN_CHAT_PROMPT
from .prompts.meal_plan_modification_chat_prompt import MEAL_PLAN_MODIFICATION_CHAT_PROMPT
from .prompts.dessert_recipe_agent_prompt import DESSERT_RECIPE_AGENT_PROMPT

class MealPlanAdvisor:
    def __init__(self, llm, retriever, conversation_memory, previous_meal_plan_memory, agent_executor):
        self.llm = llm
        self.retriever = retriever
        self.agent_executor = agent_executor
        self.conversation_memory = conversation_memory
        self.previous_meal_plan_memory = previous_meal_plan_memory
        
    def generate_meal_plan(self, user_name, dietary_restrictions, likes, dislikes):
        recipe_generation_chain = (
            {
            "user_name": lambda x: x["user_name"],
            "dietary_restrictions": lambda x: x["dietary_restrictions"],
            "likes": lambda x: x["likes"],
            "dislikes": lambda x: x["dislikes"],
            "chat_history": lambda x: self.conversation_memory.load_memory_variables(x).get("chat_history", []),
            "context": lambda x: self.retriever.invoke(f"recipes for {x['dietary_restrictions']} that include {x['likes']} but not {x['dislikes']}")}
            | MEAL_PLAN_CHAT_PROMPT
            | self.llm
        )
        user_input = {"user_name": user_name, "dietary_restrictions": dietary_restrictions, "likes": likes, "dislikes": dislikes}
        response = recipe_generation_chain.invoke(user_input)

        self.previous_meal_plan_memory.clear()
        self.previous_meal_plan_memory.save_context({"input": f"Meal plan"}, {"output": response.content})
        # history = previous_meal_plan_memory.load_memory_variables({})['previous_meal_plan']
        # print("Full meal plan history:", history)
        
        self.conversation_memory.save_context(user_input, {"answer": response.content})
        # overall_history = conversation_memory_summary.load_memory_variables({})['chat_history']
        # print("overall_history plan history:", overall_history)
        return response.content

    def generate_modified_meal_plan(self, user_name, modification_request, dietary_restrictions, likes, dislikes):
        previous_meal_plan = self.previous_meal_plan_memory.load_memory_variables({})['previous_meal_plan']
        if not previous_meal_plan:
            return "No previous previous_meal_plan found to modify."

        improved_recipe_chain = (
        {
            "user_name": lambda x: x["user_name"],
            "previous_meal_plan": lambda x: x["previous_meal_plan"],
            "modification_request": lambda x: x["modification_request"],
            "dietary_restrictions": lambda x: x["dietary_restrictions"],
            "likes": lambda x: x["likes"],
            "dislikes": lambda x: x["dislikes"],
            "chat_history": lambda x: self.conversation_memory.load_memory_variables(x).get("chat_history", []),
        }
        | MEAL_PLAN_MODIFICATION_CHAT_PROMPT
        | self.llm
    )

        user_input = {"user_name": user_name, "dietary_restrictions": dietary_restrictions, "likes": likes, "dislikes": dislikes,
                    "previous_meal_plan": previous_meal_plan, "modification_request": modification_request}
        response = improved_recipe_chain.invoke(user_input)

        self.previous_meal_plan_memory.save_context({"input": f"Meal plan"}, {"output": response.content})
        # history = previous_meal_plan_memory.load_memory_variables({})['previous_meal_plan']
        # print("Full meal plan history:", history)
        
        self.conversation_memory.save_context(user_input, {"answer": response.content})
        # overall_history = conversation_memory_summary.load_memory_variables({})['chat_history']
        # print("overall_history plan history:", overall_history)

        return response.content

    def generate_dessert_recipe(self, dietary_restrictions, likes, dislikes):
        try:
            search_query = DESSERT_RECIPE_AGENT_PROMPT.format(
                            dietary_restrictions=dietary_restrictions,
                            likes=likes,
                            dislikes=dislikes
                        )
            response = self.agent_executor.invoke({"input": search_query})

            return response['output']
        except Exception as e:
            return f"Error during web search: {e}"

