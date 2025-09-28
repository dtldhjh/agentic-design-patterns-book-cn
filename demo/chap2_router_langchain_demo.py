'''
Description: 
version: 
Author: hjh
Date: 2025-09-26 21:26:51
LastEditors: hjh
LastEditTime: 2025-09-27 22:03:08
'''
# This code is licensed under the MIT License.
# See the LICENSE file in the repository for the full license text.
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("router_langchain.log"),  # 输出到文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

# --- Configuration ---
# Ensure your API key environment variable is set (e.g., GOOGLE_API_KEY)
try:
   llm = ChatOpenAI(model = "Qwen/Qwen3-Next-80B-A3B-Instruct",
                    api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
                    base_url=os.getenv("MODELSCOPE_BASE_URL"),                  
                    temperature=0)
   logging.info("Language model initialized successfully")
except Exception as e:
   print(f"Error initializing language model: {e}")
   llm = None
# --- Define Specialist Chains ---
# Each chain represents a specific action the router can delegate to.
# 1. Booking Handler Chain
booking_chain = (
   RunnablePassthrough()  # Takes the input as-is
   | ChatPromptTemplate.from_messages([
       ("system", "You are a booking assistant. You can help users book flights and hotels. "
                  "Provide a helpful summary of the booking process and what information you would need."),
       ("user", "{input}")
   ])
   | llm
   | StrOutputParser()
)
def log_booking_handler(input):
   logging.info("\n--- DELEGATING TO BOOKING HANDLER ---")
   return input

booking_handler = RunnableLambda(log_booking_handler) | booking_chain

# 2. Info Handler Chain
info_chain = (
   RunnablePassthrough()
   | ChatPromptTemplate.from_messages([
       ("system", "You are an information assistant. Answer the user's question based on your general knowledge. "
                  "If you don't know the answer, suggest they seek a more specialized source."),
       ("user", "{input}")
   ])
   | llm
   | StrOutputParser()
)
def log_info_handler(input):
   logging.info("\n--- DELEGATING TO INFO HANDLER ---")
   return input

info_handler = RunnableLambda(log_info_handler) | info_chain

# 3. Unclear Request Handler Chain
unclear_chain = (
   RunnablePassthrough()
   | ChatPromptTemplate.from_messages([
       ("system", "The user's request is unclear. Politely ask them to rephrase or provide more details."),
       ("user", "{input}")
   ])
   | llm
   | StrOutputParser()
)
def log_unclear_handler(input):
   logging.info("\n--- HANDLING UNCLEAR REQUEST ---")
   return input

unclear_handler = RunnableLambda(log_unclear_handler) | unclear_chain
# --- Define Coordinator Router Chain (equivalent to ADK coordinator's instruction) ---
# This chain decides which handler to delegate to.
coordinator_router_prompt = ChatPromptTemplate.from_messages([
   ("system", """Analyze the user's request and determine which specialist handler should process it.
    - If the request is related to booking flights or hotels, 
      output 'booker'.
    - For all other general information questions, output 'info'.
    - If the request is unclear or doesn't fit either category, 
      output 'unclear'.
    ONLY output one word: 'booker', 'info', or 'unclear'."""),
   ("user", "{request}")
])
if llm:
   coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()
# --- Define the Delegation Logic (equivalent to ADK's Auto-Flow based on sub_agents) ---
# Use RunnableBranch to route based on the router chain's output.
# Define the branches for the RunnableBranch
branches = {
   "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),
   "info": RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])),
   "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])),
}
# Create the RunnableBranch. It takes the output of the router chain
# and routes the original input ('request') to the corresponding handler.
delegation_branch = RunnableBranch(
   (lambda x: x['decision'].strip() == 'booker', branches["booker"]), # Added .strip()
   (lambda x: x['decision'].strip() == 'info', branches["info"]),     # Added .strip()
   branches["unclear"] # Default branch for 'unclear' or any other output
)
# Combine the router chain and the delegation branch into a single runnable
# The router chain's output ('decision') is passed along with the original input ('request')
# to the delegation_branch.
coordinator_agent = {
   "decision": coordinator_router_chain,
   "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output']) # Extract the final output
# --- Define the Delegation Logic (equivalent to ADK's Auto-Flow based on sub_agents) ---
# Use RunnableBranch to route based on the router chain's output.
# Define the branches for the RunnableBranch
branches = {
   "booker": RunnablePassthrough.assign(output=lambda x: booking_handler.invoke(x['request']['request'])),
   "info": RunnablePassthrough.assign(output=lambda x: info_handler.invoke(x['request']['request'])),
   "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler.invoke(x['request']['request'])),
}
# Create the RunnableBranch. It takes the output of the router chain
# and routes the original input ('request') to the corresponding handler.
delegation_branch = RunnableBranch(
   (lambda x: x['decision'].strip() == 'booker', branches["booker"]), # Added .strip()
   (lambda x: x['decision'].strip() == 'info', branches["info"]),     # Added .strip()
   branches["unclear"] # Default branch for 'unclear' or any other output
)
# Combine the router chain and the delegation branch into a single runnable
# The router chain's output ('decision') is passed along with the original input ('request')
# to the delegation_branch.
coordinator_agent = {
   "decision": coordinator_router_chain,
   "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output']) # Extract the final output

# --- Main Execution ---
if __name__ == "__main__":
   if llm is None:
       logging.error("Skipping execution due to LLM initialization failure.")
   else:
       logging.info("--- Running with a booking request ---")
       result_a = coordinator_agent.invoke({"request": "Book me a flight to London."})
       logging.info(f"Final Result A: {result_a}")

       logging.info("\n--- Running with an info request ---")
       result_b = coordinator_agent.invoke({"request": "What is the capital of Italy?"})
       logging.info(f"Final Result B: {result_b}")

       logging.info("\n--- Running with an unclear request ---")
       result_c = coordinator_agent.invoke({"request": "Tell me about quantum physics."})
       logging.info(f"Final Result C: {result_c}")