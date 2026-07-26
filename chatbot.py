from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()

model=ChatGroq(model="openai/gpt-oss-120b") #chat model

summarize_llm=ChatGroq(model="openai/gpt-oss-120b") #summary model

print("="*60)
print("                     Hello Welcome!          ")
print("="*60)

#Chat memory
messages=[]

print("\nType 'exit' to Exit")
while True:
    prompt= input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt.lower()=="exit":
        print("Thank you, I Hope it was a good conversation!")
        break
    response=model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot : ",response.content)

    ## Summarize to save the memory
    if len(messages)>=10:
        summary_prompt="Summarize the following meassages into small paragraph: \n\n"
        for msg in messages:
            if isinstance(msg,HumanMessage):
                summary_prompt+=f"User: {msg.content}\n"
            else:
                summary_prompt+=f"Bot: {msg.content}\n"   

        summarize_response=summarize_llm.invoke(summary_prompt)
        messages.clear()
        messages.append(SystemMessage(content=summarize_response.content)) #Conversation History

        print(messages)

