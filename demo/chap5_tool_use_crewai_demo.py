'''
Description: 
version: 
Author: hjh
Date: 2025-09-27 21:06:40
LastEditors: hjh
LastEditTime: 2025-09-27 21:40:47
'''
import os
from crewai import Agent, Task, Crew,LLM
from crewai.tools import tool
import logging
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())
from langsmith import traceable

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "tool_use_agent_demo"
# --- Best Practice: Configure Logging ---
# A basic logging setup helps in debugging and tracking the crew's execution.
# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_crew.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

# --- Set up your API Key ---
# For production, it's recommended to use a more secure method for key management
# like environment variables loaded at runtime or a secret manager.
#
# Set the environment variable for your chosen LLM provider (e.g., OPENAI_API_KEY)
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
# --- 1. Refactored Tool: Returns Clean Data ---
# The tool now returns raw data (a float) or raises a standard Python error.
# This makes it more reusable and forces the agent to handle outcomes properly.
@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
   """
   Fetches the latest simulated stock price for a given stock ticker symbol.
   Returns the price as a float. Raises a ValueError if the ticker is not found.
   """
   logging.info(f"Tool Call: get_stock_price for ticker '{ticker}'")
   simulated_prices = {
       "AAPL": 178.99,
       "GOOGL": 1750.30,
       "MSFT": 425.50,
   }
   price = simulated_prices.get(ticker.upper())
   if price is not None:
       return price
   else:
       # Raising a specific error is better than returning a string.
       # The agent is equipped to handle exceptions and can decide on the next action.
       raise ValueError(f"Simulated price for ticker '{ticker.upper()}' not found.")
# --- 2. Define the Agent ---
llm = LLM(
    model = "openai/Qwen/Qwen3-Next-80B-A3B-Instruct",
                    api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
                    base_url=os.getenv("MODELSCOPE_BASE_URL"),                  
                    temperature=0)
# The agent definition remains the same, but it will now leverage the improved tool.
financial_analyst_agent = Agent(
 role='Senior Financial Analyst',
 goal='Analyze stock data using provided tools and report key prices.',
 backstory="You are an experienced financial analyst adept at using data sources to find stock information. You provide clear, direct answers.",
 verbose=True,
 tools=[get_stock_price],
 # Allowing delegation can be useful, but is not necessary for this simple task.
 allow_delegation=False,
 llm=llm
)
# --- 3. Refined Task: Clearer Instructions and Error Handling ---
# The task description is more specific and guides the agent on how to react
# to both successful data retrieval and potential errors.
analyze_aapl_task = Task(
 description=(
     "What is the current simulated stock price for Apple (ticker: AAPL)? "
     "Use the 'Stock Price Lookup Tool' to find it. "
     "If the ticker is not found, you must report that you were unable to retrieve the price."
 ),
 expected_output=(
     "A single, clear sentence stating the simulated stock price for AAPL. "
     "For example: 'The simulated stock price for AAPL is $178.15.' "
     "If the price cannot be found, state that clearly."
 ),
 agent=financial_analyst_agent,
)
analyze_asds_task = Task(
 description=(
     "What is the current simulated stock price for asds (ticker: ASDS)? "
     "Use the 'Stock Price Lookup Tool' to find it. "
     "If the ticker is not found, you must report that you were unable to retrieve the price."
 ),
 expected_output=(
     "A single, clear sentence stating the simulated stock price for ASDS. "
     "For example: 'The simulated stock price for ASDS is $178.15.' "
     "If the price cannot be found, state that clearly."
 ),
 agent=financial_analyst_agent,
)
# --- 4. Formulate the Crew ---
# The crew orchestrates how the agent and task work together.
financial_crew = Crew(
 agents=[financial_analyst_agent],
 tasks=[analyze_aapl_task, analyze_asds_task],
 verbose=True # Set to False for less detailed logs in production
)

@traceable(name="kickoff_financial_crew")
def kickoff_crew():
    return financial_crew.kickoff()

# --- 5. Run the Crew within a Main Execution Block ---
# Using a __name__ == "__main__": block is a standard Python best practice.
@traceable(name="run_financial_analysis", metadata={"version": "1.0"})
def main():
   """Main function to run the crew."""
   # Check for API key before starting to avoid runtime errors.
#    if not os.environ.get("OPENAI_API_KEY"):
#        logging.error("ERROR: The OPENAI_API_KEY environment variable is not set.")
#        logging.error("Please set it before running the script.")
#        return
   logging.info("\n## Starting the Financial Crew...")
   logging.info("---------------------------------")
  
   # The kickoff method starts the execution.
   result = kickoff_crew()
   logging.info("\n---------------------------------")
   logging.info("## Crew execution finished.")
   logging.info("\nFinal Result:\n%s", result)
   return result
if __name__ == "__main__":
   main()