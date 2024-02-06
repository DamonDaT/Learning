from dotenv import load_dotenv

from langchain import hub

from langchain.agents import load_tools
from langchain.agents import AgentExecutor, create_react_agent

from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAI

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model
llm = OpenAI()

# Tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Prompt from hub.pull("hwchase17/react")
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# Prompt template
prompt_template = PromptTemplate.from_template(template)

# Create ReAct Agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

# Run Agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "最近深圳的天气如何？如果我明天要出行，请给个穿衣建议。"})
