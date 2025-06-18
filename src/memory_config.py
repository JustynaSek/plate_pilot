from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory  
from llm_config import llm_memory

conversation_memory_summary = ConversationSummaryBufferMemory(
    llm=llm_memory,
    max_token_limit=1000,
    memory_key="chat_history",
    return_messages=True,
    input_key="dietary_restrictions"
)

previous_meal_plan_memory = ConversationBufferMemory(llm=llm_memory, memory_key="previous_meal_plan", return_messages=True)