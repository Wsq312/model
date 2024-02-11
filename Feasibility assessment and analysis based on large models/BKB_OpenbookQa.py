import json
"""
将openbookqa中的知识抽取出来，保存为json文件，列表中每个元素都是
"""
def extract_knowledge(file_list):
    knowledge = []
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                fact = line.strip()  # 移除行尾的空格和换行符
                if fact:
                    knowledge.append(fact)

    return knowledge

# 抽取Additional目录下crowdsourced-facts.txt文件中的知识
additional_knowledge = extract_knowledge(['data/openbookqa/Additional/crowdsourced-facts.txt'])

# 抽取Main目录下openbook.txt文件中的知识
main_knowledge = extract_knowledge(['data/openbookqa/Main/openbook.txt'])

# 把dev_complete.jsonl、test_complete.jsonl和train_complete.jsonl文件中的fact1字段作为知识
dev_file = 'data/openbookqa/Additional/dev_complete.jsonl'
test_file = 'data/openbookqa/Additional/test_complete.jsonl'
train_file = 'data/openbookqa/Additional/train_complete.jsonl'
json_files = [dev_file, test_file, train_file]
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            fact = data['fact1']
            main_knowledge.append(fact)

all_knowledge = additional_knowledge + main_knowledge

# 保存为json文件
with open('data/openbookqa-knowledge.json', 'w', encoding='utf-8') as f:
    json.dump(all_knowledge, f, ensure_ascii=False, indent=4)
