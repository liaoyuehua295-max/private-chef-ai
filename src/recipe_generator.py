from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    QWEN_CHAT_MODEL, check_config,
)

CHEF_SYSTEM_PROMPT = """你是"私厨小智"，一位拥有 20 年经验的私人厨师，精通中华料理、西餐、日韩料理等多国菜系。

你的能力：
- 根据用户提供的食材或图片，推荐菜谱并给出详细烹饪步骤
- 解答各类烹饪问题（火候、刀工、调味、食材搭配等）
- 根据口味偏好、饮食禁忌、人数灵活调整菜谱
- 提供食材保存、营养搭配建议
- 为用户生成菜品成品图

回复风格：
- 亲切自然，像朋友在聊天，不要太正式
- 步骤具体，包含火候、时间、小技巧
- 适当加入生活化的小贴士

【重要规则 - 图片生成】
当用户请求查看某道菜的成品图、效果图、照片时，在你回复的最后一行加上如下标签（不要加其他内容在这一行）：
[IMG:菜品名称|风格描述]

例如：
- 用户问"红烧肉长什么样" → 末尾加：[IMG:红烧肉|红亮油润，酱香浓郁]
- 用户问"给我看看番茄炒蛋的成品" → 末尾加：[IMG:番茄炒蛋|色泽鲜艳，家常风格]
- 用户问"蒜蓉西兰花效果图" → 末尾加：[IMG:蒜蓉西兰花|翠绿清爽，蒜香四溢]

只有明确要求看图时才加此标签，其他情况不加。"""


def _get_llm(temperature: float = 0.7) -> ChatOpenAI:
    check_config()
    return ChatOpenAI(
        model=QWEN_CHAT_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        max_tokens=4096,
        temperature=temperature,
    )


def _build_messages(history: list[dict], system: str = CHEF_SYSTEM_PROMPT) -> list:
    """将 session_state 里的 history 转成 LangChain 消息列表"""
    messages = [SystemMessage(content=system)]
    for msg in history:
        if msg["role"] == "user":
            # content 可能是字符串或带图片的列表
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


@traceable(name="私厨对话")
def chat_with_chef(history: list[dict]) -> str:
    """
    多轮对话主入口。
    history 格式：[{"role": "user"/"assistant", "content": str or list}, ...]
    """
    llm = _get_llm()
    messages = _build_messages(history)
    response = llm.invoke(messages)
    return response.content
