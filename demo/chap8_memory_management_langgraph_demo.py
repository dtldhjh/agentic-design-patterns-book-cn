'''
Description: 
version: 
Author: hjh
Date: 2025-10-06 11:20:07
LastEditors: hjh
LastEditTime: 2025-10-08 11:22:14
'''
# short term memory
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
import os
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
llm = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"),
        base_url="https://api-inference.modelscope.cn/v1/",
        temperature=0
    )

# Define the function that calls the model
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    # Update message history with response:
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "abc123"}}
# query = "Hi! I'm Bob."

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()  # output contains all messages in state

# query = "What's my name?"

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# query = "What's my name?"
# config = {"configurable": {"thread_id": "abc234"}}

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# long term memory
from langgraph.store.memory import InMemoryStore
# A placeholder for a real embedding function
def embed(texts: list[str]) -> list[list[float]]:
   # In a real application, use a proper embedding model
   return [[1.0, 2.0] for _ in texts]
# Initialize an in-memory store. For production, use a database-backed store.
store = InMemoryStore(index={"embed": embed, "dims": 2})
# Define a namespace for a specific user and application context
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)
# 1. Put a memory into the store
store.put(
   namespace,
   "a-memory",  # The key for this memory
   {
       "rules": [
           "User likes short, direct language",
           "User only speaks English & python",
       ],
       "my-key": "my-value",
       "foo": "bar",
   },
)
# 2. Get the memory by its namespace and key
item = store.get(namespace, "a-memory")
print("Retrieved Item:", item)
# 3. Search for memories within the namespace, filtering by content
# and sorting by vector similarity to the query.
items = store.search(
   namespace,
   filter={"my-key": "my-value","foo": "none"},
   query="language preferences"
)
print("Search Results:", items)