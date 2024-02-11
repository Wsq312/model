import json
import random
import pickle
from util import *
from tqdm import tqdm


# 根据问题得到k个知识
def get_k_knowledge(query, vector, k=5):
    docs = vector.similarity_search(query, k)
    return docs


faiss = load_embedding()
dev_file = 'data/openbookqa/Additional/dev_complete.jsonl'
test_file = 'data/openbookqa/Additional/test_complete.jsonl'
train_file = 'data/openbookqa/Additional/train_complete.jsonl'
json_files = [dev_file, test_file, train_file]

main_knowledge = []

for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            main_knowledge.append(data)

count = 0
random.shuffle(main_knowledge)
print(len(main_knowledge))

knowledge_docs_dict = {}  # 创建一个新的字典用于存储每个知识和对应的docs
for knowledge in tqdm(main_knowledge[:100]):
    print(knowledge)
    fact = knowledge["fact1"]
    stem = knowledge["question"]["stem"]
    docs = get_k_knowledge(stem, faiss, k=100)
    print(docs)

    # 将 docs 转换为你所需要的格式
    docs_dict = {doc.metadata['编号']: doc.page_content for doc in docs}
    print(docs_dict)
    knowledge_docs_dict[str(knowledge)] = docs_dict  # 注意，这里假设docs_dict是可序列化的
    for doc in docs:
        content = doc.page_content
        if fact in content:
            count += 1
            break

print("信息抽取占比:{}".format(count / 100))

# 将knowledge_docs_dict存储到磁盘上
with open('knowledge_docs_dict2.pkl', 'wb') as f:
    pickle.dump(knowledge_docs_dict, f)
