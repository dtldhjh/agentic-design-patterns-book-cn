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
from crewai import Agent, Task, Crew, Process, LLM
import logging
# Load environment variables from .env file for security
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())
# --- Best Practice: Configure Logging ---
# A basic logging setup helps in debugging and tracking the crew's execution.
# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("planning_crew.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)
# 1. Explicitly define the language model for clarity
llm = LLM(
    model = "openai/Qwen/Qwen3-Next-80B-A3B-Instruct",
                    api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
                    base_url=os.getenv("MODELSCOPE_BASE_URL"),                  
                    temperature=0.7)
# 2. Define a clear and focused agent
planner_writer_agent = Agent(
   role='Article Planner and Writer',
   goal='Plan and then write a concise, engaging summary on a specified topic.',
   backstory=(
       'You are an expert technical writer and content strategist. '
       'Your strength lies in creating a clear, actionable plan before writing, '
       'ensuring the final summary is both informative and easy to digest.'
   ),
   verbose=True,
   allow_delegation=False,
   llm=llm # Assign the specific LLM to the agent
)
# 3. Define a task with a more structured and specific expected output
topic = "The importance of Reinforcement Learning in AI"
high_level_task = Task(
   description=(
       f"1. Create a bullet-point plan for a summary on the topic: '{topic}'.\n"
       f"2. Write the summary based on your plan, keeping it around 200 words."
   ),
   expected_output=(
       "A final report containing two distinct sections:\n\n"
       "### Plan\n"
       "- A bulleted list outlining the main points of the summary.\n\n"
       "### Summary\n"
       "- A concise and well-structured summary of the topic."
   ),
   agent=planner_writer_agent,
)
# Create the crew with a clear process
crew = Crew(
   agents=[planner_writer_agent],
   tasks=[high_level_task],
   process=Process.sequential,
)
# Execute the task
logging.info("## Running the planning and writing task ##")
result = crew.kickoff()
logging.info("\n\n---\n## Task Result ##\n---")
logging.info(result)