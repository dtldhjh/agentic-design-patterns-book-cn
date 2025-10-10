'''
Description: 
version: 
Author: hjh
Date: 2025-10-10 11:23:48
LastEditors: hjh
LastEditTime: 2025-10-10 11:38:49
'''
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv,find_dotenv 
import os
import asyncio
import json
from google.adk.runners import Runner
from google.genai import types


load_dotenv(find_dotenv())
llm = LiteLlm(model="openai/Qwen/Qwen3-Next-80B-A3B-Instruct",                    
              api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请设置相应的API密钥环境变量
                    base_url=os.getenv("MODELSCOPE_BASE_URL"),                  
                    temperature=0)

# 工具函数定义
def get_precise_location_info(address: str) -> str:
    """
    获取精确位置信息的工具函数
    
    Args:
        address: 用户提供的地址信息
        
    Returns:
        包含精确位置信息的JSON字符串
    """
    # 模拟精确位置查询结果
    precise_location_data = {
        "address": address,
        "coordinates": {"latitude": 39.9042, "longitude": 116.4074},
        "full_address": f"北京市{address}",
        "postal_code": "100000",
        "district": "北京市",
        "primary_location_failed": False
    }
    
    # 将结果存储在state中供后续agent使用
    # 在实际应用中，这可能需要通过特定的机制来实现
    return json.dumps(precise_location_data)

def get_general_area_info(city: str) -> str:
    """
    获取一般区域信息的工具函数（备用方案）
    
    Args:
        city: 城市名称
        
    Returns:
        包含一般区域信息的JSON字符串
    """
    # 模拟一般区域查询结果
    general_area_data = {
        "city": city,
        "coordinates": {"latitude": 39.9042, "longitude": 116.4074},
        "province": "北京市",
        "population": "2100万",
        "area_km2": 16410,
        "primary_location_failed": False
    }
    
    return json.dumps(general_area_data)

# 创建工具对象
get_precise_location_tool = FunctionTool(get_precise_location_info)
get_general_area_tool = FunctionTool(get_general_area_info)

# Agent 1: Tries the primary tool. Its focus is narrow and clear.
primary_handler = Agent(
   name="primary_handler",
   model=llm,
   instruction="""
Your job is to get precise location information.
Use the get_precise_location_info tool with the user's provided address.
   """,
   tools=[get_precise_location_tool]
)
# Agent 2: Acts as the fallback handler, checking state to decide its action.
fallback_handler = Agent(
   name="fallback_handler",
   model=llm,
   instruction="""
Check if the primary location lookup failed by looking at state["primary_location_failed"].
- If it is True, extract the city from the user's original query and use the get_general_area_info tool.
- If it is False, do nothing.
   """,
   tools=[get_general_area_tool]
)
# Agent 3: Presents the final result from the state.
response_agent = Agent(
   name="response_agent",
   model=llm,
   instruction="""
Review the location information stored in state["location_result"].
Present this information clearly and concisely to the user.
If state["location_result"] does not exist or is empty, apologize that you could not retrieve the location.
   """,
   tools=[] # This agent only reasons over the final state.
)
# The SequentialAgent ensures the handlers run in a guaranteed order.
robust_location_agent = SequentialAgent(
   name="robust_location_agent",
   sub_agents=[primary_handler, fallback_handler, response_agent]
)

# 应用和会话配置
APP_NAME = "robust_location_agent_app"
USER_ID = "user1234"
SESSION_ID = "session1234"

async def run_robust_location_agent(user_request: str):
    """运行 robust_location_agent 来处理位置查询"""
    # 创建会话服务和会话
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    
    # 创建 runner
    runner = Runner(agent=robust_location_agent, app_name=APP_NAME, session_service=session_service)
    
    # 准备用户输入
    content = types.Content(
        role='user', 
        parts=[types.Part(text=user_request)]
    )
    
    # 运行 agent
    events = runner.run(
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        new_message=content
    )
    
    # 获取最终响应
    final_result = ""
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            final_result = event.content.parts[0].text
            print(f"Agent Final Response: {final_result}")
            
    return final_result

# 工具函数使用演示
def demo_tools():
    """演示工具函数的基本用法"""
    print("=== 工具函数演示 ===")
    
    # 演示精确位置查询工具
    print("\n1. 演示 get_precise_location_info 工具:")
    address = "朝阳区某某街道123号"
    precise_result = get_precise_location_info(address)
    print(f"   输入地址: {address}")
    print(f"   输出结果: {precise_result}")
    
    # 解析并展示结果
    parsed_result = json.loads(precise_result)
    print(f"   解析结果:")
    print(f"     - 完整地址: {parsed_result['full_address']}")
    print(f"     - 坐标: {parsed_result['coordinates']}")
    print(f"     - 邮政编码: {parsed_result['postal_code']}")
    
    # 演示一般区域查询工具
    print("\n2. 演示 get_general_area_info 工具:")
    city = "北京"
    general_result = get_general_area_info(city)
    print(f"   输入城市: {city}")
    print(f"   输出结果: {general_result}")
    
    # 解析并展示结果
    parsed_result = json.loads(general_result)
    print(f"   解析结果:")
    print(f"     - 省份: {parsed_result['province']}")
    print(f"     - 人口: {parsed_result['population']}")
    print(f"     - 面积: {parsed_result['area_km2']} 平方公里")

# 示例调用
async def main():
    # 首先运行工具函数演示
    demo_tools()
    
    print("\n=== SequentialAgent 调用演示 ===")
    try:
        result = await run_robust_location_agent("请提供北京市朝阳区某某街道的精确位置信息")
        print(f"最终结果: {result}")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())