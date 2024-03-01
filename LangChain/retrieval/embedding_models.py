from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    CASE 1:
        OpenAIEmbeddings.
"""

# Embeddings model
embeddings_model = OpenAIEmbeddings()

# Documents
embedded_docs = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)

# Query
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
