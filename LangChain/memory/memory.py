from dotenv import load_dotenv

from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)

from langchain_openai import OpenAI

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Model
llm = OpenAI()

# Conversation chain
conversation_chain = ConversationChain(llm=llm)

# Round One
round_1 = conversation_chain.invoke({"input": "你好"})

# Round Two
round_2 = conversation_chain.invoke({"input": "简单描述下龙珠是一部什么样的动漫"})

# Round Three
round_3 = conversation_chain.invoke({"input": "卡卡罗特是超级赛亚人吗"})

# Buffer
buffer = conversation_chain.memory.buffer

"""
    CASE 1:
        ConversationBufferMemory (default).
"""

# Conversation chain
conversation_chain = ConversationChain(llm=llm)

"""
    CASE 2:
        ConversationBufferWindowMemory.
"""

# Conversation chain
conversation_chain = ConversationChain(llm=llm, memory=ConversationBufferWindowMemory(k=1))

"""
    CASE 3:
        ConversationSummaryMemory.
"""

# Conversation chain
conversation_chain = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))

"""
    CASE 4:
        ConversationSummaryBufferMemory.
"""

# Conversation chain
conversation_chain = ConversationChain(llm=llm, memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=300))
