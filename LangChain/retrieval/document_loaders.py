from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, ArxivLoader, UnstructuredURLLoader

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    CASE 1:
        TextLoader. (The simplest loader reads in a file as text and places it all into one document)
"""

text_loader = TextLoader(file_path='../../Data/txt/state_of_the_union.txt')
text_docs = text_loader.load()

"""
    CASE 2:
        ArxivLoader. (Help searching papers on Arxiv)
"""

arxiv_loader = ArxivLoader(query='2005.14165', load_max_docs=5)
arxiv_docs = arxiv_loader.load()
metadata = arxiv_docs[0].metadata

"""
    CASE 3:
        UnstructuredURLLoader. (Parse web page data)
"""

unstructured_url_loader = UnstructuredURLLoader(urls=['https://react-lm.github.io/'], mode='elements')
url_docs = unstructured_url_loader.load()
metadata = url_docs[0].metadata
page_content = url_docs[0].page_content
