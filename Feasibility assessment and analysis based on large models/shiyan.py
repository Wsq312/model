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
def get_k_knowledge(query,vector,k=5):  # 定义一个函数，输入问题和向量，输出k个最相关的知识
    docs=vector.similarity_search(query,k)  # 使用输入的向量对问题进行相似度搜索，找出最相似的k个知识
    return docs

faiss=load_embedding()  # 加载词嵌入模型

#读取数据，先仅用Additional的内容
# 把dev_complete.jsonl、test_complete.jsonl和train_complete.jsonl文件中的fact1字段作为知识

dev_file = 'data/openbookqa/Additional/dev_complete.jsonl'  # 开发集的数据文件路径
test_file = 'data/openbookqa/Additional/test_complete.jsonl'  # 测试集的数据文件路径
train_file = 'data/openbookqa/Additional/train_complete.jsonl'  # 训练集的数据文件路径
json_files = [dev_file, test_file, train_file]  # 将这三个路径存入一个列表
main_knowledge=[]  # 初始化主要知识列表
for file in json_files:  # 遍历每个文件
    with open(file, 'r', encoding='utf-8') as f:  # 打开文件
        for line in f:  # 遍历文件中的每一行
            data = json.loads(line)  # 将每一行的数据加载为json格式
            main_knowledge.append(data)  # 将加载后的数据添加到主要知识列表
count=0  # 初始化计数器
random.shuffle(main_knowledge)  # 对主要知识列表进行随机打乱
print(len(main_knowledge))  # 打印主要知识列表的长度
for knowledge in tqdm(main_knowledge[:1000]):  # 对前100个知识进行遍历
    print(knowledge)
    fact=knowledge["fact1"]  # 获取每个知识的fact1字段
    stem=knowledge["question"]["stem"]  # 获取每个知识的问题stem字段
    docs=get_k_knowledge(stem,faiss,k=100)  # 对每个问题使用get_k_knowledge函数，获取最相关的100个知识
    print(docs)
    for doc in docs:  # 遍历这100个知识
        content=doc.page_content  # 获取每个知识的页面内容
        if fact in content:  # 如果fact1字段在页面内容中
            count+=1  # 计数器加1
            break  # 跳出当前循环
print("信息抽取占比:{}".format(count/100))  # 打印信息抽取占比，即有多少问题下面是对代码的注释：
