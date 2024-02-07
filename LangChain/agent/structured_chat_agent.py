from dotenv import load_dotenv

from langchain import hub

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser

from langchain_openai import ChatOpenAI

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model
llm = ChatOpenAI(model_name="gpt-4")

# Tools
sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

# Chat prompt template
chat_prompt_template = hub.pull("hwchase17/structured-chat-agent")

# Create Structured Chat Agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=chat_prompt_template)

# Run Agent
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
agent_executor.invoke({"input": "What are the headers on python.langchain.com?"})
