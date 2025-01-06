
import getpass
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set API keys
with open("keys.txt", "r") as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace or newlines
        if line and "=" in line:  # Check if line is not empty and contains an '='
            key, value = line.split("=", 1)  # Split at the first '='
            os.environ[key] = value  # Set the environment variable
# Test print the keys
print("LANGCHAIN_API_KEY:", os.environ.get("LANGCHAIN_API_KEY"))
print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")

#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#if not os.environ.get("LANGCHAIN API KEY"):
#    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass(prompt="Enter LANGCHAIN API KEY: ") # this will prompt the user to enter the API w/o it being displayed

file_path = "./cv_yuhala.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

#print(len(docs))

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "What is my name?"
)

print(results[0])

results = vector_store.similarity_search(
    "Where did I do my PhD and what was it about?"
)

print(results[0])