from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Text loader
text_loader = TextLoader(file_path='../../Data/txt/state_of_the_union.txt')
text_docs = text_loader.load()

# Splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
text_docs = text_splitter.split_documents(text_docs)

# Embeddings model
embeddings_model = OpenAIEmbeddings()

# Query
query = "What did the president say about Ketanji Brown Jackson"

"""
    CASE DB: Chroma.
"""

# Database
chroma_db = Chroma.from_documents(documents=text_docs, embedding=embeddings_model)

"""
    CASE DB: Faiss.
"""

# Database
faiss_db = FAISS.from_documents(documents=text_docs, embedding=embeddings_model)

# Persistent storage faiss DB
faiss_db.save_local("../../Data/faiss_index")

# Load faiss DB
new_faiss_db = FAISS.load_local("../../Data/faiss_index", embeddings=embeddings_model)

"""
    CASE 1:
        Similarity Search.
"""

# Similarity search
result_docs = faiss_db.similarity_search(query)

# Similarity search by vector
result_docs = faiss_db.similarity_search_by_vector(embeddings_model.embed_query(query))

"""
    CASE 2:
        TopK Retriever.
"""

# TopK Retriever
topK_retriever = new_faiss_db.as_retriever(search_kwargs={"k": 3})
result_docs = topK_retriever.get_relevant_documents(query)

"""
    CASE 3:
        Similarity threshold.
"""

score_retriever = new_faiss_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)
result_docs = score_retriever.get_relevant_documents(query)
