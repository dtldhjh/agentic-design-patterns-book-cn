'''
Description: 
version: 
Author: hjh
Date: 2025-09-26 20:18:40
LastEditors: hjh
LastEditTime: 2025-09-27 22:29:47
'''
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
# To enhance security, please load environment variables from the.env file
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())
# Ensure your OPENAI_API_KEY is set in the.env file
# Initialize the language model (it is recommended to use ChatOpenAI)
llm = ChatOpenAI(api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
    base_url=os.getenv("MODELSCOPE_BASE_URL"),
    model = "Qwen/Qwen3-Next-80B-A3B-Instruct",
    temperature=0)

# --- Prompt 1: Extract Information ---
prompt_extract = ChatPromptTemplate.from_template(
   "Extract the technical specifications from the following text:\n\n{text_input}"
)
# --- Prompt 2: Transform to JSON ---
prompt_transform = ChatPromptTemplate.from_template(
   "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
)
# --- Build the Chain using LCEL ---
# The StrOutputParser() converts the LLM's message output to a simple string.
extraction_chain = prompt_extract | llm | StrOutputParser()
# The full chain passes the output of the extraction chain into the 'specifications'
# variable for the transformation prompt.
full_chain = (
   {"specifications": extraction_chain}
   | prompt_transform
   | llm
   | StrOutputParser()
)
# --- Run the Chain ---
input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."
# Execute the chain with the input text dictionary.
final_result = full_chain.invoke({"text_input": input_text})
print("\n--- Final JSON Output ---")
print(final_result)