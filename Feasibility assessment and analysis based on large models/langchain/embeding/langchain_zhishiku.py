from langchain.document_loaders import UnstructuredFileLoader
import os
os.environ['http_proxy'] = 'http://127.0.0.1:10812'
os.environ['https_proxy'] = 'http://127.0.0.1:10812'

def load_pdf(pdf_path):
  loader = UnstructuredFileLoader(pdf_path)
  docs = loader.load()
  return docs

docs = load_pdf("222.pdf")
for doc in docs:
    print(doc.page_content)

print (f'You have {len(docs)} document(s) in your data')
print (f'There are {len(docs[0].page_content)} characters in your document')

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
print (f'Now you have {len(split_docs)} documents')
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
OPENAI_API_KEY = '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'
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(temperature=0.6, openai_api_key=OPENAI_API_KEY)
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
input_docs = split_docs
chain.run(input_documents=input_docs)