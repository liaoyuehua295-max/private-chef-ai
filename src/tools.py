from langchain_core.tools import tool
from langsmith import traceable

# 存放 Agent 运行过程中生成的图片（dish_name → PIL Image）
generated_images: dict = {}


@tool
def generate_recipe(ingredients: str, preferences: str = "家常口味，2人份") -> str:
    """
    根据食材列表生成详细菜谱。
    ingredients: 食材列表，逗号分隔，如"番茄、鸡蛋、葱"
    preferences: 口味偏好和人数，如"清淡，3人份"
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, QWEN_CHAT_MODEL

    llm = ChatOpenAI(
        model=QWEN_CHAT_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        max_tokens=2048,
        temperature=0.7,
    )
    messages = [
        SystemMessage(content="你是一位专业私人厨师，请根据食材生成详细菜谱，包含步骤、火候、时间和小贴士。"),
        HumanMessage(content=f"食材：{ingredients}\n偏好：{preferences}\n请推荐2道菜并给出详细做法。"),
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def get_nutrition_info(ingredients: str) -> str:
    """
    查询食材或菜品的营养价值信息。
    ingredients: 食材或菜品名称，如"番茄"或"红烧肉"
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, QWEN_CHAT_MODEL

    llm = ChatOpenAI(
        model=QWEN_CHAT_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        max_tokens=512,
        temperature=0.3,
    )
    messages = [
        SystemMessage(content="你是营养师，简洁介绍食材的主要营养成分和健康功效，100字以内。"),
        HumanMessage(content=f"请介绍{ingredients}的营养价值。"),
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def suggest_shopping(current_ingredients: str) -> str:
    """
    根据现有食材，建议补充采购哪些食材能做出更多菜。
    current_ingredients: 当前已有食材，逗号分隔
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, QWEN_CHAT_MODEL

    llm = ChatOpenAI(
        model=QWEN_CHAT_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        max_tokens=512,
        temperature=0.5,
    )
    messages = [
        SystemMessage(content="你是私人厨师，根据现有食材建议再买2-3样食材，能搭配出更多菜，要具体说明搭配原因。"),
        HumanMessage(content=f"我现在有：{current_ingredients}，建议我再买什么？"),
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def create_dish_image(dish_name: str, style: str = "精致摆盘，美食摄影") -> str:
    """
    生成菜品成品效果图。调用此工具后图片会直接展示给用户。
    dish_name: 菜品名称，如"番茄炒蛋"
    style: 图片风格描述，如"家常风格"或"餐厅摆盘"
    """
    from src.image_gen import generate_dish_image as _gen
    try:
        img = _gen(dish_name, style)
        # 存入全局字典，app.py 读取后展示
        generated_images[dish_name] = img
        return f"IMAGE_READY:{dish_name}"
    except Exception as e:
        return f"图片生成失败：{e}"


@tool
def estimate_cooking_time(dish_name: str) -> str:
    """
    用户输入菜名，返回备料时间、烹饪时间、总时长
    dish_name: 菜品名称，如"番茄炒蛋"
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, QWEN_CHAT_MODEL

    llm = ChatOpenAI(
        model=QWEN_CHAT_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        max_tokens=512,
        temperature=0.5,
    )
    messages = [
        SystemMessage(content="你是私人厨师，根据菜名，能说明备料时间，烹饪时间，以及总时长"),
        HumanMessage(content=f"请告诉我：{dish_name}的烹饪时间"),
    ]
    response = llm.invoke(messages)
    return response.content








# 工具列表，提供给 Agent 使用
CHEF_TOOLS = [
    generate_recipe,
    get_nutrition_info,
    suggest_shopping,
    create_dish_image,
    estimate_cooking_time,
]
