from util import *
from tqdm import tqdm
import random

"""
本代码用于测试和评估模型信息筛选的能力，对于输入的问题的和已知知识，观察模型是否可以
很好的识别和找到对应知识
1.根据问题，找到k个最相关的知识
2.模型再进行一次筛选
3.将数据进行保存
"""

#根据问题得到k个知识
def get_k_knowledge(query,vector,k=5):
    docs=vector.similarity_search(query,k)
    return docs

faiss=load_embedding()

#读取数据，先仅用Additional的内容

# 把dev_complete.jsonl、test_complete.jsonl和train_complete.jsonl文件中的fact1字段作为知识
dev_file = 'data/openbookqa/Additional/dev_complete.jsonl'
test_file = 'data/openbookqa/Additional/test_complete.jsonl'
train_file = 'data/openbookqa/Additional/train_complete.jsonl'
json_files = [dev_file, test_file, train_file]
main_knowledge=[]
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            main_knowledge.append(data)
count=0
random.shuffle(main_knowledge)
print(len(main_knowledge))
for knowledge in tqdm(main_knowledge[:100]):
    fact=knowledge["fact1"]
    stem=knowledge["question"]["stem"]
    docs=get_k_knowledge(stem,faiss,k=100)
    for doc in docs:
        content=doc.page_content
        if fact in content:
            count+=1
            break
print("信息抽取占比:{}".format(count/100))
