from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Initialize environment variables from .env file
load_dotenv(verbose=True)

"""
    CASE 1:
        RecursiveCharacterTextSplitter. (Text)
"""

file_text = """
    So you're a man who has everything, and nothing.
    Sometimes you gotta run before you can walk.
    I just finally know what I have to do. And I know in my heart that it's right.
"""

# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True
)

# Documents
text_docs = text_splitter.create_documents([file_text], metadatas=[{"document": 1}])

"""
    CASE 2:
        RecursiveCharacterTextSplitter. (Language)
"""

html_text = """
<!DOCTYPE html>
<html>
    <head>
        <title>🦜️🔗 LangChain</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            h1 {
                color: darkblue;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>🦜️🔗 LangChain</h1>
            <p>⚡ Building applications with LLMs through composability ⚡</p>
        </div>
        <div>
            As an open source project in a rapidly developing field, we are extremely open to contributions.
        </div>
    </body>
</html>
"""

# Splitter
language_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, chunk_size=60, chunk_overlap=0
)

# Documents
language_docs = language_splitter.create_documents([html_text])
