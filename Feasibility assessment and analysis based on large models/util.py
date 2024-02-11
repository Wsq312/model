import json
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

"""
加载embedding向量
"""

os.environ['http_proxy'] = 'http://127.0.0.1:10812'
os.environ['https_proxy'] = 'http://127.0.0.1:10812'
os.environ["OPENAI_API_KEY"] = "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"
def load_embedding(file_path="data/openbookqa-triple-knowledge.json",embeddings=OpenAIEmbeddings()):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        triple_knowledge = json.load(json_file)
    print(len(triple_knowledge))
    texts = [element[0] for element in triple_knowledge]
    texts_embeddings = [element[1] for element in triple_knowledge]
    metadata = [element[2] for element in triple_knowledge]
    text_embedding_pairs = list(zip(texts, texts_embeddings))
    faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings, metadata)
    return faiss
