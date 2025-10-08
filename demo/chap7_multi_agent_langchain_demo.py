'''
Description: 
version: 
Author: hjh
Date: 2025-09-28 22:05:37
LastEditors: hjh
LastEditTime: 2025-10-05 17:10:19
'''
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv,find_dotenv 

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_agent_langchain.log"),
        logging.StreamHandler()
    ]
)

def main():
    """
    Initializes and runs the AI agents for content creation using LangChain.
    """
    # 设置LangSmith追踪
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "multi_agent_demo"
    
    # Define the language model to use.
    llm = ChatOpenAI(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"),
        base_url="https://api-inference.modelscope.cn/v1/",
        temperature=0
    )
    
    # Define prompts for each agent
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an experienced research analyst with a knack for identifying key trends and synthesizing information."),
        ("user", "Research the top 3 emerging trends in Artificial Intelligence in 2024-2025. Focus on practical applications and potential impact.")
    ])
    
    writing_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a skilled writer who can translate complex technical topics into accessible content."),
        ("user", "Write a 500-word blog post based on the following research findings. The post should be engaging and easy for a general audience to understand.\n\nResearch Findings:\n{research_findings}")
    ])
    
    # Create chains for each agent
    research_chain = research_prompt | llm | StrOutputParser()
    writing_chain = writing_prompt | llm | StrOutputParser()
    
    # Execute the research agent
    logging.info("## Running the research agent... ##")
    try:
        research_findings = research_chain.invoke({})
        logging.info("\n------------------\n")
        logging.info("## Research Findings ##")
        logging.info(research_findings)
        
        # Execute the writing agent with research findings as context
        logging.info("\n## Running the writing agent... ##")
        blog_post = writing_chain.invoke({"research_findings": research_findings})
        logging.info("\n------------------\n")
        logging.info("## Final Blog Post ##")
        logging.info(blog_post)
        
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()