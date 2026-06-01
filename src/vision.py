import base64
import json
import re
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    QWEN_VISION_MODEL, check_config,
)


def _encode_image(image: Image.Image, max_size: int = 1024) -> tuple[str, str]:
    """压缩并 base64 编码图片"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"


def _get_vision_llm() -> ChatOpenAI:
    check_config()
    return ChatOpenAI(
        model=QWEN_VISION_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        max_tokens=1024,
    )


@traceable(name="识别食材")
def identify_ingredients(image: Image.Image) -> dict:
    """
    用通义千问 VL 识别图片中的食材。
    返回：{ingredients, states, scene, raw}
    """
    llm = _get_vision_llm()
    encoded, mime = _encode_image(image)

    prompt = """你是一位专业的厨师助手，请仔细观察这张图片。

请完成以下任务：
1. 识别图片中所有可见的食材（蔬菜、肉类、海鲜、调料、食品等）
2. 描述每种食材的状态（新鲜/熟制/腌制/冷冻等）

请严格按照以下 JSON 格式输出，不要添加任何其他文字：
{
  "ingredients": ["食材1", "食材2"],
  "states": {"食材1": "状态描述"},
  "scene": "整体场景描述"
}"""

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            },
            {"type": "text", "text": prompt},
        ]
    )

    response = llm.invoke([message])
    raw = response.content.strip()

    # 提取 JSON
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {
                "ingredients": result.get("ingredients", []),
                "states": result.get("states", {}),
                "scene": result.get("scene", ""),
                "raw": raw,
            }
        except json.JSONDecodeError:
            pass

    # 回退：逐行解析
    lines = [l.strip().lstrip("-•·").strip() for l in raw.splitlines() if l.strip()]
    return {"ingredients": lines, "states": {}, "scene": "", "raw": raw}
