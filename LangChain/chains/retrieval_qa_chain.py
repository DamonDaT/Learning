from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    CASE:
        RetrievalQA. (Combined with faiss vector database)
"""

# Create documents
text_loader = TextLoader(file_path='../../Data/txt/state_of_the_union.txt')
text_docs = text_loader.load()
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
text_docs = text_splitter.split_documents(text_docs)

# Model
embeddings_model = OpenAIEmbeddings()
llm_model = ChatOpenAI(model_name="gpt-4", temperature=0.8)

# Faiss
faiss_db = FAISS.from_documents(documents=text_docs, embedding=embeddings_model)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    retriever=faiss_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})
)

result = qa_chain.invoke({"query": "What did the president say about Ketanji Brown Jackson"})
