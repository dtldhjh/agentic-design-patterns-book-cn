'''
Description: 
version: 
Author: hjh
Date: 2025-10-05 16:30:38
LastEditors: hjh
LastEditTime: 2025-10-06 11:20:07
'''

# ChatMessageHistory: Manual Memory Management
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
   ChatPromptTemplate,
   MessagesPlaceholder,
   SystemMessagePromptTemplate,
   HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils import convert_to_secret_str
from dotenv import load_dotenv, find_dotenv 
load_dotenv(find_dotenv())
# 1. Define Chat Model and Prompt
llm = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"),
        base_url="https://api-inference.modelscope.cn/v1/",
    )
prompt = ChatPromptTemplate(
   messages=[
       SystemMessagePromptTemplate.from_template("You are a friendly assistant."),
       MessagesPlaceholder(variable_name="chat_history"),
       HumanMessagePromptTemplate.from_template("{question}")
   ]
)
# 2. Configure Memory
# return_messages=True is essential for chat models
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 3. Build the Chain using LCEL (LangChain Expression Language)
# Replacing deprecated LLMChain with recommended approach
chain = prompt | llm | StrOutputParser()

# 4. Run the Conversation
response = chain.invoke({
    "question": "Hi, I'm Jane.",
    "chat_history": memory.chat_memory.messages
})
print(response)

# Save the interaction to memory
memory.chat_memory.add_user_message("Hi, I'm Jane.")
memory.chat_memory.add_ai_message(response)

response = chain.invoke({
    "question": "Do you remember my name?",
    "chat_history": memory.chat_memory.messages
})
print(response)