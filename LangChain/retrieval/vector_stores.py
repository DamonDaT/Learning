from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Text loader
text_loader = TextLoader(file_path=r'D:\Data\XXX\YYY.txt')
text_docs = text_loader.load()

# Splitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
text_docs = text_splitter.split_documents(text_docs)

# Embeddings model
embeddings_model = OpenAIEmbeddings()

"""
    CASE 1:
        Chroma.
"""

# Database
chroma_db = Chroma.from_documents(text_docs, embeddings_model)

# Similarity search
query = "What did the president say about Ketanji Brown Jackson"
query_docs = chroma_db.similarity_search(query)

# Similarity search by vector
query = "What did the president say about Ketanji Brown Jackson"
query_docs = chroma_db.similarity_search_by_vector(query)
