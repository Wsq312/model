from langchain.chat_models import ChatOpenAI
from util import *
from langchain import PromptTemplate
from langchain.chains import LLMChain
import json
import time
os.environ['http_proxy'] = 'http://127.0.0.1:10792'
os.environ['https_proxy'] = 'http://127.0.0.1:10792'

openai_api_key = "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"

"""
本代码的用于评估大模型在依靠支持事实时模型的性能，实现流程如下：
1.循环读取数据
2.对于每个数据的问题，首先根据问题从知识库提取潜在的支持事实
3.大模型根据问题，筛选出能够解决问题的支持事实
4.模型根据支持事实解决问题
"""


#将支持事实和编号进行返回
def get_fact_and_number(docs):
    fact_list=""
    for doc in docs:
        number="number-"+str(doc.metadata["编号"])+":"
        content=doc.page_content
        fact_list+=number
        fact_list+=content
        fact_list+="\n"
    return fact_list
#根据问题提取知识，并转换格式
def extract_knowledge(query,vector,k=20):
    docs = vector.similarity_search(query, k)
    fact_list = get_fact_and_number(docs)
    return fact_list



#第一阶段，抽取支持事实的模板
template = """Please find supporting facts that can answer the following questions. If there are any, please provide the supporting facts. If not, please return empty. Here is an example for reference:
Question：Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as
Optional：A.Deep sea animals B.fish  C.Long Sea Fish D.Far Sea Animals
Potential Supported facts: number-340:Examples of deep sea animals are angler fish and frilled sharks 
number-5747:deep sea animals live deep in the ocean 
number-524:Lophiiformes include anglerfish 
number-4281:squids live deep in the ocean 
number-4112:sharks have gills 
number-5694:clams live at the bottom of the ocean 
number-2446:flounder are aquatic animals 
number-1483:angler fish use lights to attract prey 
number-4110:sharks are carnivores 
number-5304:a fish lives in water
Supported facts:["Examples of deep sea animals are angler fish and frilled sharks","deep sea animals live deep in the ocean"]

Now, please output the supporting facts based on the questions, strictly following the above output format. If there are no supporting facts, return empty:
question:{question}
Optional:{Optional}
Potential Supported facts:{Potential_facts}"""

#第二阶段模板
template2 = """Please answer the following questions. If there are supporting facts, provide an interpretation of the answer based on the supporting facts. Here is an example for reference:
Question：Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as
Optional：A.Deep sea animals B.fish  C.Long Sea Fish D.Far Sea Animals
Supported facts:["Examples of deep sea animals are angler fish and frilled sharks","deep sea animals live deep in the ocean"]
Answer: A
Explanation: Frilled sharks and angler fish are known as deep sea animals. This is supported by the fact that they live deep beneath the surface of the ocean. Deep sea animals are specifically adapted to survive in the extreme conditions of the deep ocean, where there is high pressure, low temperatures, and limited light penetration. The examples of angler fish and frilled sharks demonstrate the characteristics of deep sea animals. Therefore, option A, "Deep sea animals," is the most appropriate answer.

Remember, this is a multiple-choice question.Now, please output the correct answers based on the questions, strictly following the above output format:
question:{question}
Optional:{Optional}
Potential Supported facts:{facts}"""

prompt_template = PromptTemplate(input_variables=["question", "Optional","Potential_facts"], template=template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.6,max_tokens=2048)
chain = LLMChain(llm=llm, prompt=prompt_template)


prompt_template_answer = PromptTemplate(input_variables=["question", "Optional","facts"], template=template2)
llm1 = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.6,max_tokens=2048)
chain_anwer = LLMChain(llm=llm1, prompt=prompt_template_answer)
if __name__=="__main__":
    #读取数据
    # 把dev_complete.jsonl、test_complete.jsonl和train_complete.jsonl文件中的fact1字段作为知识
    dev_file = 'data/openbookqa/Additional/dev_complete.jsonl'
    test_file = 'data/openbookqa/Additional/test_complete.jsonl'
    train_file = 'data/openbookqa/Additional/train_complete.jsonl'
    json_files = [test_file]
    main_knowledge=[]
    Faiss=load_embedding()
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                main_knowledge.append(data)
    #循环遍历数据
    fact_match_counter=[]
    for knowlede in main_knowledge:
        #获取信息
        question=knowlede["question"]["stem"]
        print("问题:{}".format(question))
        answerKey=knowlede["answerKey"]

        True_fact=knowlede["fact1"]


        #获得选项
        optional=""
        for option_dict in knowlede["question"]["choices"]:
            optional += option_dict["label"]+"."
            optional += option_dict["text"] + "\t"
        print("选项:{}".format(optional))
        print("答案:{}".format(answerKey))
        print("真实支撑:{}".format(True_fact))

        #抽取相关事实
        Potential_facts=extract_knowledge(question,Faiss,k=20)

        #根据问题采用抽取支持事实
        # print("ChatGPT Q:{}".format(prompt_template.format(question=question, Optional=optional,Potential_facts=Potential_facts)))
        res=chain.run({
            "question":question,
            "Optional":optional,
            "Potential_facts":Potential_facts
        })
        res=res.replace("Supported facts:","")
        if "空" in res and len(res)<10:
            fact_match_counter.append(0)
            continue
        print("预测支撑:{}".format(res))
        # print("True fact:{}".format(True_fact))
        surport_facts = json.loads(res)

        answer1=chain_anwer.run({
            "question":question,
            "Optional":optional,
            "facts":res
        })
        time.sleep(1)
        print("预测答案:{}".format(answer1))
        print("\n")

        if True_fact in surport_facts:
            fact_match_counter.append(1)
        else:
            fact_match_counter.append(0)
        time.sleep(1)


