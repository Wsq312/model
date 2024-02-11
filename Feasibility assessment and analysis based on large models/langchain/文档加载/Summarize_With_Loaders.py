import os
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredPowerPointLoader, ReadTheDocsLoader, PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
os.environ['http_proxy'] = 'http://127.0.0.1:10812'
os.environ['https_proxy'] = 'http://127.0.0.1:10812'
OPENAI_API_KEY = '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'
def summarize_docs(docs, doc_url):
    print (f'You have {len(docs)} document(s) in your {doc_url} data')
    print (f'There are {len(docs[0].page_content)} characters in your document')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    print (f'You have {len(split_docs)} split document(s)')
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="text-davinci-003")
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
    response = ""
    with get_openai_callback() as cb:
        response = chain.run(input_documents=split_docs)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
    print(response)
    return response

url = "https://mil.news.sina.com.cn/2023-05-10/doc-imythsmx9833111.shtml"
print(UnstructuredURLLoader(urls = [url]).load(), url)
summarize_docs(UnstructuredURLLoader(urls = [url]).load(), url)

#
# loader = UnstructuredPowerPointLoader("Web3-intro.pptx")
# response = summarize_docs(loader.load(), "Web3-intro.pptx")
# print(response)
#
#
# loader = ReadTheDocsLoader("langchain")
# summarize_docs(loader.load(), "langchain")
#
#
# loader = PyPDFLoader("tsla-20221231-gen.pdf")
# pages = loader.load_and_split()
# summarize_docs(pages[:10], "tsla-20221231-gen.pdf")