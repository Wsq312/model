from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os


os.environ['http_proxy'] = 'http://127.0.0.1:10812'
os.environ['https_proxy'] = 'http://127.0.0.1:10812'
OPENAI_API_KEY = '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'
# Load the vectorstore from disk
persist_directory = 'chroma_storage'
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

query = "如何利用Solidity实现插入排序？"
docs = vectordb.similarity_search(query)
print(len(docs))
print(docs)


import chromadb
from chromadb.config import Settings
client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory))
client.list_collections()

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)

query = "Solidity的特点"
docs = vectorstore.similarity_search(query, 3, include_metadata=True)
# print(len(docs))
# print(docs[0])

a=chain.run(input_documents=docs, question=query)
print(a)