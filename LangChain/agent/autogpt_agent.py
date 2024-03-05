import faiss
from dotenv import load_dotenv

from langchain_experimental.autonomous_agents import AutoGPT
from langchain.agents.tools import Tool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore import InMemoryDocstore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Google search api
search = SerpAPIWrapper()

tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events."
    ),
    WriteFileTool(),
    ReadFileTool()
]

# Model
embeddings_model = OpenAIEmbeddings()
llm_model = ChatOpenAI(model_name="gpt-4", temperature=0, verbose=True)

# OpenAI embedding dimension
embedding_size = 1536
# 使用 Faiss 的 IndexFlatL2 索引
index = faiss.IndexFlatL2(embedding_size)
# 实例化 Faiss 向量数据库
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# Agent
agent = AutoGPT.from_llm_and_tools(
    ai_name="Jarvis",
    ai_role="Assistant",
    tools=tools,
    llm=llm_model,
    memory=vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8}),  # 实例化 Faiss 的 VectorStoreRetriever
)

# Turn on detailed record
agent.chain.verbose = True

agent.run(["2023年成都大运会，中国金牌数是多少"])
