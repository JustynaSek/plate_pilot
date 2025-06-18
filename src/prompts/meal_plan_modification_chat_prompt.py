from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

MEAL_PLAN_MODIFICATION_INSTRUCTIONS_HUMAN_TEMPLATE = '''
    You have already provided a meal plan for a {user_name} with dietary restrictions: {dietary_restrictions}, considering taste preferences: likes: {likes}, dislikes: {dislikes}.
    The user has requested a modification to the meal plan: {modification_request}. Here is meal plan:
    {previous_meal_plan}
    Please provide an updated meal plan, ensuring to include the user's preferences and the context of the original recipe.'''

MEAL_PLAN_MODIFICATION_INSTRUCTIONS_SYSTEM_TEMPLATE = """
    You are a helpful dietician. Your goal is to provide an updated meal plan based on the user's dietary restrictions 
    and food preferences, using attached recepie. Use only the context provided."
"""
MEAL_PLAN_MODIFICATION_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(MEAL_PLAN_MODIFICATION_INSTRUCTIONS_SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),  
    HumanMessagePromptTemplate.from_template(MEAL_PLAN_MODIFICATION_INSTRUCTIONS_HUMAN_TEMPLATE)
])