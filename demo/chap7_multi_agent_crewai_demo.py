'''
Description: 
version: 
Author: hjh
Date: 2025-09-28 22:05:37
LastEditors: hjh
LastEditTime: 2025-09-28 22:08:36
'''
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
# def setup_environment():
#    """Loads environment variables and checks for the required API key."""
#    load_dotenv()
#    if not os.getenv("GOOGLE_API_KEY"):
#        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
def main():
   """
   Initializes and runs the AI crew for content creation using the latest Gemini model.
   """
#    setup_environment()
   # Define the language model to use.
   # Updated to a model from the Gemini 2.0 series for better performance and features.
   # For cutting-edge (preview) capabilities, you could use "gemini-2.5-flash".
   llm = LLM(
    model = "openai/Qwen/Qwen3-Next-80B-A3B-Instruct",
                    api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
                    base_url=os.getenv("MODELSCOPE_BASE_URL"),                  
                    temperature=0)
   # Define Agents with specific roles and goals
   researcher = Agent(
       role='Senior Research Analyst',
       goal='Find and summarize the latest trends in AI.',
       backstory="You are an experienced research analyst with a knack for identifying key trends and synthesizing information.",
       verbose=True,
       allow_delegation=False,
       llm=llm
   )
   writer = Agent(
       role='Technical Content Writer',
       goal='Write a clear and engaging blog post based on research findings.',
       backstory="You are a skilled writer who can translate complex technical topics into accessible content.",
       verbose=True,
       allow_delegation=False,
       llm=llm
   )
   # Define Tasks for the agents
   research_task = Task(
       description="Research the top 3 emerging trends in Artificial Intelligence in 2024-2025. Focus on practical applications and potential impact.",
       expected_output="A detailed summary of the top 3 AI trends, including key points and sources.",
       agent=researcher,
   )
   writing_task = Task(
       description="Write a 500-word blog post based on the research findings. The post should be engaging and easy for a general audience to understand.",
       expected_output="A complete 500-word blog post about the latest AI trends.",
       agent=writer,
       context=[research_task],
   )
   # Create the Crew
   blog_creation_crew = Crew(
       agents=[researcher, writer],
       tasks=[research_task, writing_task],
       process=Process.sequential,
       verbose=True # Set verbosity for detailed crew execution logs
   )
   # Execute the Crew
   print("## Running the blog creation crew with Gemini 2.0 Flash... ##")
   try:
       result = blog_creation_crew.kickoff()
       print("\n------------------\n")
       print("## Crew Final Output ##")
       print(result)
   except Exception as e:
       print(f"\nAn unexpected error occurred: {e}")
if __name__ == "__main__":
   main()