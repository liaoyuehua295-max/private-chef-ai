from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langsmith import traceable

from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    QWEN_CHAT_MODEL, check_config,
)
from src.tools import CHEF_TOOLS

AGENT_SYSTEM_PROMPT = """你是"私厨小智"，一位拥有 20 年经验的私人厨师 AI 助手。

你拥有以下工具，根据用户需求自主决定调用哪些工具、调用几次：
- generate_recipe：根据食材生成详细菜谱
- get_nutrition_info：查询食材或菜品的营养价值
- suggest_shopping：建议补充采购的食材
- create_dish_image：生成菜品成品效果图
- estimate_cooking_time：用户输入菜名，返回备料时间、烹饪时间、总时长

工作原则：
- 先思考用户真正需要什么，再决定调用哪些工具
- 多个需求可以连续调用多个工具
- 【重要】工具返回的内容必须完整地写进你的最终回复中，用户只能看到你的回复，看不到工具的原始输出
- 用亲切自然的语气组织内容，不要说"我调用了工具"这类话"""


def _get_llm() -> ChatOpenAI:
    check_config()
    return ChatOpenAI(
        model=QWEN_CHAT_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        max_tokens=4096,
        temperature=0.7,
    )


def _build_messages(history: list[dict], user_input: str) -> list:
    messages = []
    for msg in history[:-1]:
        if msg["role"] == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                messages.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_input))
    return messages


def _make_agent():
    llm = _get_llm()
    return create_agent(
        model=llm,
        tools=CHEF_TOOLS,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


@traceable(name="私厨Agent对话")
def run_agent(user_input: str, history: list[dict]) -> str:
    """运行 Agent，返回最终回复文字。"""
    check_config()
    agent = _make_agent()
    messages = _build_messages(history, user_input)
    result = agent.invoke({"messages": messages})

    output_messages = result.get("messages", [])
    for msg in reversed(output_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "抱歉，我没能生成回复，请再试一次。"


@traceable(name="私厨Agent带进度")
def run_agent_with_steps(user_input: str, history: list[dict]) -> tuple[str, list[str]]:
    """
    运行 Agent，同时返回调用了哪些工具。
    返回：(最终回复文字, [调用的工具名列表])
    """
    check_config()
    agent = _make_agent()
    messages = _build_messages(history, user_input)

    # 用 invoke 拿完整结果，比 stream 更可靠
    result = agent.invoke({"messages": messages})
    output_messages = result.get("messages", [])

    steps = []      # 记录工具调用顺序
    final_reply = ""

    for msg in output_messages:
        if isinstance(msg, AIMessage):
            # 有 tool_calls → AI 决定调工具，记录工具名
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if tool_name:
                        steps.append(tool_name)
            # 有 content → 可能是最终回复（取最后一条）
            elif msg.content:
                final_reply = msg.content

    if not final_reply:
        final_reply = "抱歉，我没能生成回复，请再试一次。"

    return final_reply, steps
