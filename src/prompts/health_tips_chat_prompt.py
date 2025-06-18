from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

HEALTH_TIPS_SYSTEM_TEMPLATE = """
        You are a health adviser for programmers and web developers.
        Your tips are witty, encouraging, and actionable, using intelligent tech analogies and a friendly,
        knowledgeable tone. Never be rude or condescending."""

HEALTH_TIPS_HUMAN_TEMPLATE = """My name is {user_name}. {user_full_query}"""

HEALTH_TIPS_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(HEALTH_TIPS_SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HEALTH_TIPS_HUMAN_TEMPLATE),
    ]
)

QWEN_SYSTEM_MESSAGE = "You are a health adviser for programmers and web developers. Your tips are witty, encouraging, and actionable, using intelligent tech analogies and a friendly, knowledgeable tone. Never be rude or condescending."
