from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))


llm_summarize = init_chat_model("openai/gpt-oss-20b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))

REDIS_URL=os.environ.get("REDIS_URL")
REDIS_KEY_PREFIX=f"{input("Enter User ID: ")}:"

MAX_MESSAGES_BEFORE_SUMMARY=8

def get_redis_history(session_id:str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix=REDIS_KEY_PREFIX,
        #ttl=3300 # time chat will stay in memory
    )

def get_chain_with_history():
    prompt=ChatPromptTemplate.from_messages(
        [

    ("system","You are a helpful assistant, who answer user question"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}")

        ]
    )

    chain= prompt | llm

    chain_with_history= RunnableWithMessageHistory(
        chain,get_redis_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    return chain_with_history


if __name__=='__main__':
    config={"configurable":{"session_id":"456"}}
    chain_with_history=get_chain_with_history()
    while True:
        message = input('You: ')
        if message.lower()=="exit":
            break
        response=chain_with_history.invoke(
            {"input":message},
            config=config)
        
        print("Assistant: ",response.content)