from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

MEAL_PLAN_INSTRUCTION_HUMAN_TEMPLATE = """
    Concise one-day meal plan (breakfast, lunch, dinner) for {user_name} with simple, quick recipes for a person with dietary restrictions:{dietary_restrictions},
    considering taste preferences: likes: {likes}, dislikes: {dislikes}
    and adult portions. Provide brief description, main ingredients, and short instructions for each meal.
    Be so nice to format is as a list with bullet points.
    To answer the user's request, use only the following context about suitable recipes:
    {context}
"""
MEAL_PLAN_INSTRUCTION_SYSTEM_TEMPLATE = """
    You are a  helpful dietician. Your goal is to provide a one-day meal plan based on the user's health condition and food preferences,
    using retrieved recipes. Use only the context provided. 
    Do not make anything up if you haven't been provided with relevant context
    Answer the user's request in a witty and informative manner."
"""

MEAL_PLAN_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(MEAL_PLAN_INSTRUCTION_SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"), 
    HumanMessagePromptTemplate.from_template(MEAL_PLAN_INSTRUCTION_HUMAN_TEMPLATE)
])