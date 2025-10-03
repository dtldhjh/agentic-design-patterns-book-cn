'''
Description: 
version: 
Author: hjh
Date: 2025-09-28 22:05:37
LastEditors: hjh
LastEditTime: 2025-09-28 22:08:36
'''
import os
import logging
from typing import Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_agent_langgraph.log"),
        logging.StreamHandler()
    ]
)

# Define state
class AgentState(TypedDict):
    research_findings: str
    blog_post: str
    messages: Annotated[list, add_messages]

def research_node(state: AgentState):
    """Research agent that identifies AI trends"""
    # 设置LangSmith追踪
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "multi_agent_demo"
    
    llm = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"),
        base_url="https://api-inference.modelscope.cn/v1/",
        temperature=0
    )
    
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an experienced research analyst with a knack for identifying key trends and synthesizing information."),
        ("user", "Research the top 3 emerging trends in Artificial Intelligence in 2024-2025. Focus on practical applications and potential impact.")
    ])
    
    research_chain = research_prompt | llm | StrOutputParser()
    research_findings = research_chain.invoke({})
    
    logging.info("## Research Findings ##")
    logging.info(research_findings)
    
    return {"research_findings": research_findings}

def writing_node(state: AgentState):
    """Writing agent that creates blog post based on research"""
    llm = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"),
        base_url="https://api-inference.modelscope.cn/v1/",
        temperature=0
    )
    
    writing_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a skilled writer who can translate complex technical topics into accessible content."),
        ("user", "Write a 500-word blog post based on the following research findings. The post should be engaging and easy for a general audience to understand.\n\nResearch Findings:\n{research_findings}")
    ])
    
    writing_chain = writing_prompt | llm | StrOutputParser()
    blog_post = writing_chain.invoke({"research_findings": state["research_findings"]})
    
    logging.info("## Final Blog Post ##")
    logging.info(blog_post)
    
    return {"blog_post": blog_post}

def should_continue(state: AgentState) -> Literal["writer", "end"]:
    """Determine if we should continue to writing or end"""
    if state.get("research_findings"):
        return "writer"
    return "end"

def main():
    """
    Initializes and runs the AI agents for content creation using LangGraph.
    """
    # Build the graph
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("researcher", research_node)
    builder.add_node("writer", writing_node)
    
    # Add edges
    builder.add_edge(START, "researcher")
    builder.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "writer": "writer",
            "end": END,
        },
    )
    builder.add_edge("writer", END)
    
    # Compile the graph
    graph = builder.compile()
    
    # Execute the graph
    logging.info("## Running the blog creation agents with LangGraph... ##")
    try:
        result = graph.invoke({})
        logging.info("\n------------------\n")
        logging.info("## Crew Final Output ##")
        logging.info(f"Research Findings:\n{result.get('research_findings', 'None')}")
        logging.info(f"\nBlog Post:\n{result.get('blog_post', 'None')}")
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    load_dotenv()
    main()