from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.callbacks import get_openai_callback
import os
os.environ['http_proxy'] = 'http://127.0.0.1:10792'
os.environ['https_proxy'] = 'http://127.0.0.1:10792'


OPENAI_API_KEY = ""
QUERIES = [
    "My interest is to explore the options of scaling Ethereum",
    "Could you please elaborate more on sharding? Try to use at least 1000 words.",
    "What are the cons of sharding?"
    "What should I learn if I decide to work on Ethereum?",
    "What are the most important skills for a blockchain developer?",
    "I have some basic understanding of smart contracts. Other than the basic programming skills, what else should I learn? I know NFT is pretty popular, what should I be capable of doing with NFT?",
    "Opensea is one of the most popular NFT marketplace. What's its architecture? How can I build something similar?",
    "How can I run such a marketplace on Ethereum? What's the cost of running such a marketplace? I would like to know the typical business model of such a marketplace.",
    "In terms of marketing, as more and more NFT collections are published on Opensea, how can my marketplace compete with them? What's the potential opportunity for me to win the battle?",
    "What are the most popular NFT collections on Opensea? What's the typical price of a NFT collection? How can I get a NFT collection on Opensea?"
]



def track_tokens_usage(chain, query, tokens, requests):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Total tokens: {cb.total_tokens}')
        print(f'Requests: {cb.successful_requests}')
        tokens.append(cb.total_tokens)
        requests.append(cb.successful_requests)
    return result



def start_conversation(llm, queries, memory_type):
    chain = ConversationChain(llm=llm, memory=memory_type)
    tokens = []
    requests = []
    for query in queries:
        print(f'Query: {query}')
        result = track_tokens_usage(chain, query, tokens, requests)
        print(f'Result: {result}')
        print('')

    return tokens, requests



tokens1, requests1 = start_conversation(OpenAI(
    temperature=0,
	openai_api_key=OPENAI_API_KEY,
	model_name="text-davinci-003"
), QUERIES, ConversationBufferMemory())



llm = OpenAI(
    temperature=0,
	openai_api_key=OPENAI_API_KEY,
	model_name="text-davinci-003"
)

(tokens2, requests2) = start_conversation(llm, QUERIES, ConversationSummaryMemory(llm=llm))


#k=1,记录最近一次的对话内容
(tokens3, requests3) = start_conversation(OpenAI(
    temperature=0,
	openai_api_key=OPENAI_API_KEY,
	model_name="text-davinci-003"
), QUERIES, ConversationBufferWindowMemory(k=1))




llm = OpenAI(
    temperature=0,
	openai_api_key=OPENAI_API_KEY,
	model_name="text-davinci-003"
)
#如果没有超过600，就直接记忆，如果超过了600就进行摘要记忆
(tokens4, requests4) = start_conversation(llm, QUERIES, ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=600
))

import matplotlib.pyplot as plt

xs = range(1, len(QUERIES) + 1)
plt.plot(xs, tokens1, label='Buffer Memory')
plt.plot(xs, tokens2, label='Summary Memory')
plt.plot(xs, tokens3, label='Buffer Window Memory')
plt.plot(xs, tokens4, label='Summary Buffer Memory')

plt.xlabel('Index of Queries')
plt.ylabel('Tokens')
plt.title('Tokens Usage')
plt.legend()
plt.show()