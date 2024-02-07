from dotenv import load_dotenv

from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.tools import Tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.chains import LLMMathChain

from langchain_openai import OpenAI, ChatOpenAI

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model
model = ChatOpenAI(temperature=0)

# Tool 1
search = SerpAPIWrapper()

# Tool 2
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.invoke,
        description="useful for when you need to answer questions about math"
    ),
]

# Planner
planner = load_chat_planner(model)

# Executor
executor = load_agent_executor(model, tools, verbose=True)

# Run Agent
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
agent.invoke({"input": "卡卡罗特的超级赛亚人1和超级赛亚人2的战斗力数值相差了多少?"})
