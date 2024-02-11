import json
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm
"""
将openbookqa中的知识保存为嵌入向量，保存格式为，
["知识"，嵌入向量，编码]
"""

os.environ['http_proxy'] = 'http://127.0.0.1:10812'
os.environ['https_proxy'] = 'http://127.0.0.1:10812'
os.environ["OPENAI_API_KEY"] = "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"

with open("data/openbookqa-knowledge.json", 'r', encoding='utf-8') as json_file:
    knowledge = json.load(json_file)
new_knowledge=[]
for know in tqdm(knowledge):
    if know not in new_knowledge:
        new_knowledge.append(know)
knowledge=new_knowledge
embeddings = OpenAIEmbeddings()

knowledge=knowledge
knowledge_embeddings = embeddings.embed_documents(knowledge)

triple_knowledge=[]
for i in range(len(knowledge)):
    triple_knowledge.append([knowledge[i],knowledge_embeddings[i],{"编号":i}])
# print(triple_knowledge[:][:2])
texts=[element[0] for element in triple_knowledge]
texts_embeddings=[element[1] for element in triple_knowledge]
metadata=[element[2] for element in triple_knowledge]
text_embedding_pairs = list(zip(texts, texts_embeddings))
faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings,metadata)

query=knowledge[0]
ans=faiss.similarity_search(query)

print(ans)

# 保存为json文件
with open('data/openbookqa-triple-knowledge.json', 'w', encoding='utf-8') as f:
    json.dump(triple_knowledge, f, ensure_ascii=False, indent=4)