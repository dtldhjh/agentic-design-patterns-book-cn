'''
Description: 
version: 
Author: hjh
Date: 2025-09-28 21:23:58
LastEditors: hjh
LastEditTime: 2025-09-28 21:30:36
'''
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
# Load environment variables from .env file for security
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "planning_agent_demo"

# --- Best Practice: Configure Logging ---
# A basic logging setup helps in debugging and tracking the crew's execution.
# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("planning_langchain.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)
# 1. Explicitly define the language model for clarity
llm = ChatOpenAI(
    model = "Qwen/Qwen3-Next-80B-A3B-Instruct",
    api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
    base_url="https://api-inference.modelscope.cn/v1/",                  
    temperature=0.7)

# 2. Define prompts for planning and writing
planning_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert content strategist. Create a bullet-point plan for a summary on the given topic."),
    ("user", "Create a bullet-point plan for a summary on the topic: '{topic}'.")
])

writing_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical writer. Write a concise, engaging summary based on the provided plan."),
    ("user", "Based on the following plan, write a summary around 200 words on the topic: '{topic}'.\n\nPlan:\n{plan}")
])

# 3. Create chains for planning and writing
planning_chain = planning_prompt | llm | StrOutputParser()
writing_chain = writing_prompt | llm | StrOutputParser()

# 4. Define the topic
topic = "The importance of Reinforcement Learning in AI"

# Execute the planning and writing process
logging.info("## Running the planning and writing task ##")

# First, generate the plan
plan = planning_chain.invoke({"topic": topic})
logging.info(f"\n\n---\n## Plan ##\n---\n{plan}")

# Then, write the summary based on the plan
summary = writing_chain.invoke({"topic": topic, "plan": plan})
logging.info(f"\n\n---\n## Summary ##\n---\n{summary}")

# Combine results for final output
final_result = f"### Plan\n{plan}\n\n### Summary\n{summary}"
logging.info("\n\n---\n## Task Result ##\n---")
logging.info(final_result)