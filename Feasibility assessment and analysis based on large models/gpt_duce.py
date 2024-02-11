import os
import openai
import pickle
import json
import ast
import re
import time

os.environ['http_proxy'] = 'http://127.0.0.1:10812'
os.environ['https_proxy'] = 'http://127.0.0.1:10812'

openai_api_key = "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"

class OpenaiChatModule:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def chat_with_origin_model(self, text):
        # Reinitialize the conversation history each time
        origin_model_conversation = [{"role": "system", "content": "你现在是"}]
        openai.api_key = self.openai_api_key
        text = text.replace('\n', ' ').replace('\r', '').strip()
        if len(text) == 0:
            return ""
        print(f'chatGPT Q:{text}')
        origin_model_conversation.append({"role": "user", "content": text})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=origin_model_conversation,
            max_tokens=2000,
            temperature=0.6,
            stop=None,
        )
        reply = response.choices[0].message.content
        return reply

if __name__ == '__main__':
    openai_chat_module = OpenaiChatModule(openai_api_key)
    with open('knowledge_docs_dict2.pkl', 'rb') as f:
        knowledge_docs_dict = pickle.load(f)

    answer_key_match_counter = 0
    fact_match_counter = 0
    total_counter = 0
    answers = []
    for knowledge_str, docs in knowledge_docs_dict.items():
        knowledge = ast.literal_eval(knowledge_str)
        stem = knowledge['question']['stem']
        choices =knowledge['question']['choices']
        question = stem + " 回答选项： " + str(choices) + "。回答问题，并从下面文档中找出最能支持这个答案的理由，回答格式：{'answerKey': '' ，'fact': '', 编号：''} 如果找不到证据，就把fact设置为空值，除了回答，不要说其他任何话。不要有任何解释，要求格式非常规范"
        input_text = question + str(docs)

        try:
            answer = openai_chat_module.chat_with_origin_model(input_text)
            answer = re.findall(r'\{.*\}', answer)[0]
            answer = answer.replace('，', ',').replace('编号', 'id')
            answer_dict = ast.literal_eval(answer)
            if 'answerKey' in answer_dict and answer_dict['answerKey'] == knowledge['answerKey']:
                answer_key_match_counter += 1
            if 'fact' in answer_dict and answer_dict['fact'] == knowledge['fact1']:
                fact_match_counter += 1
            total_counter += 1
            print(answer)
            print(knowledge['answerKey'])
            print(knowledge['fact1'])
            print("\n")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            answers.append(answer)
            time.sleep(1)

    answer_key_accuracy = (answer_key_match_counter / total_counter) * 100
    fact_accuracy = (fact_match_counter / total_counter) * 100

    print(f"Answer Key Match Accuracy: {answer_key_accuracy}%")
    print(f"Fact Match Accuracy: {fact_accuracy}%")

    with open('answers.pkl', 'wb') as f:
        pickle.dump(answers, f)
