'''
Description: 
version: 
Author: hjh
Date: 2025-09-27 19:44:47
LastEditors: hjh
LastEditTime: 2025-09-27 22:33:10
'''
import os, getpass
import asyncio
import nest_asyncio
from typing import List
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tool_agent.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "tool_use_agent_demo"
try:
  # A model with function/tool calling capabilities is required.
  llm = ChatOpenAI(model = "Qwen/Qwen3-Next-80B-A3B-Instruct",
                    api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
                    base_url=os.getenv("MODELSCOPE_BASE_URL"),                  
                    temperature=0)
except Exception as e:
  logging.error(f"🛑 Error initializing language model: {e}")
  llm = None
# --- Define a Tool ---
@langchain_tool
def search_information(query: str) -> str:
  """
  Provides factual information on a given topic. Use this tool to find answers to phrases
  like 'capital of France' or 'weather in London?'.
  """
  logging.info(f"\n--- 🛠️ Tool Called: search_information with query: '{query}' ---")
  # Simulate a search tool with a dictionary of predefined results.
  simulated_results = {
      "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
      "capital of france": "The capital of France is Paris.",
      "population of earth": "The estimated population of Earth is around 8 billion people.",
      "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
      "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."
  }
  result = simulated_results.get(query.lower(), simulated_results["default"])
  logging.info(f"--- TOOL RESULT: {result} ---")
  return result
tools = [search_information]
# --- Create a Tool-Calling Agent ---
if llm:
  # This prompt template requires an `agent_scratchpad` placeholder for the agent's internal steps.
  agent_prompt = ChatPromptTemplate.from_messages([
      ("system", "You are a helpful assistant."),
      ("human", "{input}"),
      ("placeholder", "{agent_scratchpad}"),
  ])
  # Create the agent, binding the LLM, tools, and prompt together.
  agent = create_tool_calling_agent(llm, tools, agent_prompt)
  # AgentExecutor is the runtime that invokes the agent and executes the chosen tools.
  # The 'tools' argument is not needed here as they are already bound to the agent.
  agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)
async def run_agent_with_tool(query: str):
  """Runs the agent with a specific query."""
  try:
      if agent_executor is None:
          raise ValueError("Agent executor is not initialized.")
      logging.info(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
      response = await agent_executor.ainvoke({"input": query})
      logging.info("\n--- ✅ Final Agent Response ---")
      logging.info(response["output"])
      return response["output"]
  except Exception as e:
      logging.error(f"\n🛑 An error occurred during agent execution: {e}")
      return None
async def main():
  """Runs all agent queries concurrently."""
  tasks = [
      run_agent_with_tool("What is the capital of France?"),
      run_agent_with_tool("What's the weather like in London?"),
      run_agent_with_tool("Tell me something about dogs.") # Should trigger the default tool response
  ]
  await asyncio.gather(*tasks)
nest_asyncio.apply()
asyncio.run(main())