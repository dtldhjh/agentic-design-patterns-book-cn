'''
Description: 
version: 
Author: hjh
Date: 2025-09-27 15:57:47
LastEditors: hjh
LastEditTime: 2025-09-27 16:13:37
'''
import os
import asyncio
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parallelization_langchain.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "parallel_agent_demo"
# --- Configuration ---
# Ensure your API key environment variable is set (e.g., OPENAI_API_KEY)
try:
   llm: Optional[ChatOpenAI] =  ChatOpenAI(model = "Qwen/Qwen3-Next-80B-A3B-Instruct",
                    api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
                    base_url=os.getenv("MODELSCOPE_BASE_URL"),                  
                    temperature=0)
   logging.info("Language model initialized successfully")
  
except Exception as e:
   logging.error(f"Error initializing language model: {e}")
   llm = None
# --- Define Independent Chains ---
# These three chains represent distinct tasks that can be executed in parallel.
summarize_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "Summarize the following topic concisely:"),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)
questions_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "Generate three interesting questions about the following topic:"),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)
terms_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)
# --- Build the Parallel + Synthesis Chain ---
# 1. Define the block of tasks to run in parallel. The results of these,
#    along with the original topic, will be fed into the next step.
map_chain = RunnableParallel(
   {
       "summary": summarize_chain,
       "questions": questions_chain,
       "key_terms": terms_chain,
       "topic": RunnablePassthrough(),  # Pass the original topic through
   }
)
# 2. Define the final synthesis prompt which will combine the parallel results.
synthesis_prompt = ChatPromptTemplate.from_messages([
   ("system", """Based on the following information:
    Summary: {summary}
    Related Questions: {questions}
    Key Terms: {key_terms}
    Synthesize a comprehensive answer."""),
   ("user", "Original topic: {topic}")
])
# 3. Construct the full chain by piping the parallel results directly
#    into the synthesis prompt, followed by the LLM and output parser.
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()
# --- Run the Chain ---
async def run_example(topic: str):
   """Run the parallel chain example with a given topic."""
   if llm is None:
       logging.error("LLM not initialized. Cannot run example.")
       return

   logging.info(f"\n--- Running Parallel LangChain Example for Topic: '{topic}' ---")
   try:
       # Invoke the chain with the input topic
       response = await parallel_chain.ainvoke({"topic": topic})
       logging.info("\n--- Final Response ---")
       logging.info(response)
   except Exception as e:
       logging.error(f"\nAn error occurred during chain execution: {e}")

if __name__ == "__main__":
   test_topic = "The history of space exploration"
   # In Python 3.7+, asyncio.run is the standard way to run an async function.
   asyncio.run(run_parallel_example(test_topic))