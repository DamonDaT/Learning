from dotenv import load_dotenv

from langchain import hub

from langchain.agents import AgentExecutor, create_self_ask_with_search_agent, Tool
from langchain_community.utilities.serpapi import SerpAPIWrapper

from langchain_openai import ChatOpenAI

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model
llm = ChatOpenAI(model_name="gpt-4")

# Tools
search_api_wrapper = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search_api_wrapper.run,
        description="useful for when you need to ask with search"
    )
]

# Chat prompt template
prompt_template = hub.pull("hwchase17/self-ask-with-search")

# Create Structured Chat Agent
agent = create_self_ask_with_search_agent(llm=llm, tools=tools, prompt=prompt_template)

# Run Agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "龙珠动漫里的主角的夫人分别叫什么名字？"})
