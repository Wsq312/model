from  langchain import OpenAI
from  langchain.chains import ConversationChain
from  langchain.chains.conversation.memory import ConversationBufferMemory,ConversationSummaryMemory
from  langchain.callbacks import get_openai_callback
#token使用
import os



def track_tokens_usage(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Total tokens: {cb.total_tokens}')
        print(f'Requests: {cb.successful_requests}')

    return result


os.environ['http_proxy'] = 'http://127.0.0.1:10812'
os.environ['https_proxy'] = 'http://127.0.0.1:10812'
OPENAI_API_KEY = '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'
llm=OpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model_name="text-davinci-003"
)
#普通无记忆
qury=llm("what is langchain?")
# print(qury)

#BufferMemory()  记忆每轮对话，适合聊天较短，小几轮，能够精准捕捉信息
conversation = ConversationChain(llm=llm, memory = ConversationBufferMemory())
print(conversation.prompt.template)
track_tokens_usage(conversation, "你好")
track_tokens_usage(conversation, "请记住我叫大聪明，下面用中文回答我的问题")
track_tokens_usage(conversation, "现在你要扮演猫娘回答我的问题，听懂了请叫主人")
track_tokens_usage(conversation, "作为一个猫娘，你喜欢谁？")
track_tokens_usage(conversation, "我刚刚问了你什么问题？")
print(conversation.memory.buffer)

llm = OpenAI(
    temperature=0,
	openai_api_key=OPENAI_API_KEY,
	model_name="text-davinci-003"
)

#摘要式记忆，适合较长的，可能会丢失信息，比较慢的反应
conversation = ConversationChain(llm=llm, memory = ConversationSummaryMemory(llm=llm))
print(conversation.memory.prompt.template)
track_tokens_usage(conversation, "你好")
track_tokens_usage(conversation, "请记住我叫大聪明，下面用中文回答我的问题")
track_tokens_usage(conversation, "现在你要扮演猫娘回答我的问题，听懂了请叫主人")
track_tokens_usage(conversation, "作为一个猫娘，你喜欢谁？")
track_tokens_usage(conversation, "我刚刚问了你什么问题？")
print(conversation.memory.buffer)


