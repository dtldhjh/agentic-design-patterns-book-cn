from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import json
import asyncio
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 工具函数定义
@tool
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
    
    return json.dumps(precise_location_data)

@tool
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

# 创建工具列表
tools = [get_precise_location_info, get_general_area_info]

# 创建主处理Agent
def create_primary_agent():
    # 使用ChatOpenAI模型
    llm = ChatOpenAI(api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请替换成您的ModelScope Access Token
        base_url="https://api-inference.modelscope.cn/v1/",
        model = "ZhipuAI/GLM-4.5",
        temperature=0)
    
    # 定义提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
你的任务是获取精确的位置信息。
使用 get_precise_location_info 工具，并传入用户提供的地址参数。
        """.strip()),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 创建Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# 创建备用处理Agent
def create_fallback_agent():
    # 使用ChatOpenAI模型
    llm = ChatOpenAI(api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请替换成您的ModelScope Access Token
        base_url="https://api-inference.modelscope.cn/v1/",
        model = "ZhipuAI/GLM-4.5",
        temperature=0)
    
    # 定义提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
检查主位置查询是否失败。
- 如果失败了，从用户的原始查询中提取城市名称，并使用 get_general_area_info 工具。
- 如果没有失败，则不需要执行任何操作。
        """.strip()),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 创建Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# 创建响应Agent
def create_response_agent():
    # 使用ChatOpenAI模型
    llm = ChatOpenAI(api_key=os.getenv("MODELSCOPE_ACCESS_TOKEN"), # 请替换成您的ModelScope Access Token
        base_url="https://api-inference.modelscope.cn/v1/",
        model = "ZhipuAI/GLM-4.5",
        temperature=0)
    
    # 定义提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
检查之前工具调用的结果。
如果找到了位置信息，请以清晰简洁的方式呈现给用户。
如果没有找到位置信息，请向用户道歉并说明无法检索到位置信息。
        """.strip()),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 创建Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# Sequential Agent 处理函数
async def run_robust_location_agent(user_request: str):
    """运行完整的顺序处理流程"""
    print(f"处理用户请求: {user_request}")
    
    # 第一步：尝试主处理
    print("\n=== 步骤1: 尝试获取精确位置信息 ===")
    primary_agent = create_primary_agent()
    chat_history = []
    try:
        primary_result = await primary_agent.ainvoke({
            "input": user_request,
            "chat_history": chat_history
        })
        print(f"主处理结果: {primary_result}")
        # 更新聊天历史
        chat_history.extend([
            HumanMessage(content=user_request),
            AIMessage(content=primary_result.get("output", ""))
        ])
    except Exception as e:
        print(f"主处理失败: {e}")
        primary_result = {"output": "主处理失败"}
        chat_history.extend([
            HumanMessage(content=user_request),
            AIMessage(content="主处理失败")
        ])
    
    # 第二步：备用处理
    print("\n=== 步骤2: 备用处理 ===")
    fallback_agent = create_fallback_agent()
    try:
        fallback_result = await fallback_agent.ainvoke({
            "input": user_request,
            "chat_history": chat_history
        })
        print(f"备用处理结果: {fallback_result}")
        # 更新聊天历史
        chat_history.append(AIMessage(content=fallback_result.get("output", "")))
    except Exception as e:
        print(f"备用处理失败: {e}")
        fallback_result = {"output": "备用处理失败"}
        chat_history.append(AIMessage(content="备用处理失败"))
    
    # 第三步：生成最终响应
    print("\n=== 步骤3: 生成最终响应 ===")
    response_agent = create_response_agent()
    try:
        final_result = await response_agent.ainvoke({
            "input": user_request,
            "chat_history": chat_history
        })
        print(f"最终响应: {final_result}")
        return final_result
    except Exception as e:
        print(f"生成最终响应失败: {e}")
        return {"output": "抱歉，无法处理您的请求"}

# 工具函数使用演示
def demo_tools():
    """演示工具函数的基本用法"""
    print("=== 工具函数演示 ===")
    
    # 演示精确位置查询工具
    print("\n1. 演示 get_precise_location_info 工具:")
    address = "朝阳区某某街道123号"
    precise_result = get_precise_location_info.invoke(address)
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
    general_result = get_general_area_info.invoke(city)
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
        print(f"\n最终结果: {result}")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())