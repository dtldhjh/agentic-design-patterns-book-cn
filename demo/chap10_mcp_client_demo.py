'''
Description: 
version: 
Author: hjh
Date: 2025-10-09 15:07:50
LastEditors: hjh
LastEditTime: 2025-10-09 22:08:44
'''
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["/Users/hejiahui/code_library/llm_agent/demo/chap10_mcp_stdio_server.py"],
            "transport": "stdio",
        },
        "weather": {
            "command": "python",
            # Replace with absolute path to your weather_server.py file
            "args": ["/Users/hejiahui/code_library/llm_agent/demo/chap10_mcp_stdio_server.py"],
            "transport": "stdio",
        },
        "greet": {
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)
async def main():
    tools = await client.get_tools()
    # print(tools)
    llm = ChatOpenAI(api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请替换成您的ModelScope Access Token
    base_url="https://api-inference.modelscope.cn/v1/",
    model = "ZhipuAI/GLM-4.5",
    temperature=0)
    agent = create_react_agent(
        llm,
        tools
    )
    # greet_response = await agent.ainvoke(
    #     {"messages": [{"role": "system", "content": "你是一个友好的助手，可以根据人们的名字向他们打招呼。使用 `greet` 工具。"},
    #                   {"role": "user", "content": "我是jane"}]}
    # )
    # print(greet_response['messages'][-1].content)
    weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)
    print(weather_response)
#     math_response = await agent.ainvoke(
#     {"messages": [{"role": "user", "content": "what's 3 + 5?"}]}
# )
#     print(math_response['messages'][-1].content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())