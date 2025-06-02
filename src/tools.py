from langchain_community.utilities import SearchApiAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
import os
from langchain import hub
from llm_config import llm_agent

searchapi_api_key = os.environ.get("SEARCHAPI_API_KEY")
tools = []
if searchapi_api_key:
    search_tool = SearchApiAPIWrapper(searchapi_api_key=searchapi_api_key)
    tools = [
        Tool(
            name="intermediate_answer",
            func=search_tool.run,
            description="useful for when you need to ask with search",
        )
    ]
else:
    tools = []
    print("SerpAPI API key not found. Web search will be disabled.")

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm_agent, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)