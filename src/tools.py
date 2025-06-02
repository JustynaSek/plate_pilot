# src/langchain_logic/core/tools.py

# --- Imports ---
# Remove: from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun # New import for DuckDuckGo
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub
from llm_config import llm_agent 

duckduckgo_search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="web_search",
        func=duckduckgo_search.run,
        description="Useful for when you need to answer questions about current events or perform general web searches, especially for recipes or dietary information. Input should be a concise query.",
    )
]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm_agent, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)