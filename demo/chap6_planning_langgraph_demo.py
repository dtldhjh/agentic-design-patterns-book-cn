'''
Description: 
version: 
Author: hjh
Date: 2025-09-28 21:23:58
LastEditors: hjh
LastEditTime: 2025-09-28 21:30:36
'''
import os
from typing import Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# Load environment variables from .env file for security
from dotenv import load_dotenv, find_dotenv 
load_dotenv(find_dotenv())

# Configure LangSmith for tracing and monitoring
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
        logging.FileHandler("planning_langgraph.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

# 1. Explicitly define the language model for clarity
llm = ChatOpenAI(
    model = "Qwen/Qwen3-Next-80B-A3B-Instruct",
    api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
    base_url="https://api-inference.modelscope.cn/v1/",                  
    temperature=0.7)

# 2. Define state
class PlanExecuteState(TypedDict):
    topic: str
    plan: str
    summary: str
    messages: Annotated[list, add_messages]

# 3. Define nodes
def plan_node(state: PlanExecuteState):
    """Generate a plan for the given topic"""
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert content strategist. Create a bullet-point plan for a summary on the given topic."),
        ("user", "Create a bullet-point plan for a summary on the topic: '{topic}'.")
    ])
    
    planning_chain = planning_prompt | llm | StrOutputParser()
    plan = planning_chain.invoke({"topic": state["topic"]})
    
    logging.info(f"Generated plan:\n{plan}")
    
    return {"plan": plan}

def execute_node(state: PlanExecuteState):
    """Execute the plan by writing a summary"""
    writing_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical writer. Write a concise, engaging summary based on the provided plan."),
        ("user", "Based on the following plan, write a summary around 200 words on the topic: '{topic}'.\n\nPlan:\n{plan}")
    ])
    
    writing_chain = writing_prompt | llm | StrOutputParser()
    summary = writing_chain.invoke({"topic": state["topic"], "plan": state["plan"]})
    
    logging.info(f"Generated summary:\n{summary}")
    
    return {"summary": summary}

# 4. Build the graph
builder = StateGraph(PlanExecuteState)

# Add nodes
builder.add_node("planner", plan_node)
builder.add_node("executor", execute_node)

# Add edges
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", END)

# Compile the graph
graph = builder.compile()

# 5. Run the graph
topic = "The importance of Reinforcement Learning in AI"
logging.info(f"## Running the planning and writing task for topic: {topic} ##")

result = graph.invoke({"topic": topic})

final_result = f"### Plan\n{result['plan']}\n\n### Summary\n{result['summary']}"
logging.info("\n\n---\n## Task Result ##\n---")
logging.info(final_result)