import uuid
from typing import Dict, Any, Optional
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event
import os
import logging
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("router_adk.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

# --- Define Tool Functions ---
# These functions simulate the actions of the specialist agents.
def booking_handler(request: str) -> str:
   """
   Handles booking requests for flights and hotels.
   Args:
       request: The user's request for a booking.
   Returns:
       A confirmation message that the booking was handled.
   """
   logging.info("-------------------------- Booking Handler Called ----------------------------")
   return f"Booking handled for request: '{request}'. This is a simulation."

def info_handler(request: str) -> str:
   """
   Handles information requests.
   Args:
       request: The user's request for information.
   Returns:
       A response message that the information request was handled.
   """
   logging.info("-------------------------- Info Handler Called ----------------------------")
   return f"Information provided for request: '{request}'. This is a simulation."

def unclear_handler(request: str) -> str:
   """Handles requests that couldn't be delegated."""
   return f"Coordinator could not delegate request: '{request}'. Please clarify."
# --- Create Tools from Functions ---
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

# Use LiteLlm to wrap the LLaMA model
llm = LiteLlm(model="openai/Qwen/Qwen3-Next-80B-A3B-Instruct",                    
              api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
                    base_url=os.getenv("MODELSCOPE_BASE_URL"),                  
                    temperature=0)
# Define specialized sub-agents equipped with their respective tools
booking_agent = Agent(
   name="Booker",
   model=llm,
   description="""A specialized agent that handles all flight 
           and hotel booking requests by calling the booking tool.""",
   tools=[booking_tool]
)
info_agent = Agent(
   name="Info",
   model=llm,
   description="""A specialized agent that provides general information
      and answers user questions by calling the info tool.""",
   tools=[info_tool]
)
# Define the parent agent with explicit delegation instructions
coordinator = Agent(
   name="Coordinator",
   model=llm,
   instruction=(
       "You are the main coordinator. Your only task is to analyze incoming user requests "
       "and delegate them to the appropriate specialist agent. Do not try to answer the user directly.\n"
       "- For any requests related to booking flights or hotels, delegate to the 'Booker' agent.\n"
       "- For all other general information questions, delegate to the 'Info' agent."
   ),
   description="A coordinator that routes user requests to the correct specialist agent.",
   # The presence of sub_agents enables LLM-driven delegation (Auto-Flow) by default.
   sub_agents=[booking_agent, info_agent]
)
# --- Execution Logic ---
async def run_coordinator(request: str) -> str:
   """Helper function to run the coordinator agent with a given request."""
   logging.info(f"\n--- Running Coordinator with request: '{request}' ---")
   
   # Session and Runner
   session_service = InMemorySessionService()
   session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
   runner = InMemoryRunner(agent=coordinator_agent, app_name=APP_NAME, session_service=session_service)
   
   content = types.Content(role='user', parts=[types.Part(text=request)])
   events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
   
   final_result = ""
   for event in events:
       if event.is_final_response():
           final_result = event.content.parts[0].text
           logging.info(f"Coordinator Final Response: {final_result}")
           
   return final_result

async def main():
   """Main function to run the routing example."""
   try:
       result_a = await run_coordinator("I want to book a flight to Paris.")
       logging.info(f"Final Output A: {result_a}")
       
       result_b = await run_coordinator("What are the visa requirements for Japan?")
       logging.info(f"Final Output B: {result_b}")
       
       result_c = await run_coordinator("What's the weather like?")
       logging.info(f"Final Output C: {result_c}")
       
       result_d = await run_coordinator("I want to book a hotel in London.")
       logging.info(f"Final Output D: {result_d}")
       
   except Exception as e:
       logging.error(f"An error occurred while processing your request: {e}")

if __name__ == "__main__":
   logging.info("--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
   logging.info("Note: This requires Google ADK installed and authenticated.")
   nest_asyncio.apply()
   asyncio.run(main())
