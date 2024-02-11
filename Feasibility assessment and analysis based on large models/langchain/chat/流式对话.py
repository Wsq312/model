from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

os.environ['http_proxy'] = 'http://127.0.0.1:10812'
os.environ['https_proxy'] = 'http://127.0.0.1:10812'
OPENAI_API_KEY = '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'
llm = OpenAI(
    temperature=0,
	openai_api_key=OPENAI_API_KEY,
	model_name="text-davinci-003",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
    streaming=True
)


response = llm("讲一个大聪明的故事.")

print(response)