import requests
from io import BytesIO
from PIL import Image
from langsmith import traceable

import dashscope
from dashscope import ImageSynthesis

from src.config import DASHSCOPE_API_KEY, check_config


@traceable(name="生成菜品图片")
def generate_dish_image(dish_name: str, style_hint: str = "") -> Image.Image:
    """
    调用通义万象文生图，返回 PIL Image。
    dish_name: 菜品名称
    style_hint: 额外风格描述，如"红亮油润"、"清淡素雅"
    """
    check_config()
    dashscope.api_key = DASHSCOPE_API_KEY

    prompt = (
        f"{dish_name}，"
        f"{style_hint + '，' if style_hint else ''}"
        "精致摆盘，美食摄影，高清特写，餐厅风格，暖光打光，食欲感强"
    )

    rsp = ImageSynthesis.call(
        model="wanx2.1-t2i-turbo",
        prompt=prompt,
        n=1,
        size="1024*1024",
    )

    if rsp.status_code != 200:
        raise RuntimeError(f"图片生成失败（{rsp.status_code}）：{rsp.message}")

    img_url = rsp.output.results[0].url
    resp = requests.get(img_url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))
