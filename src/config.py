import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "私厨AI")

# 通义千问
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "qwen-vl-max")
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-max")

# 通义千问 OpenAI 兼容接口地址
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# App
APP_TITLE = os.getenv("APP_TITLE", "私厨AI - 智能菜谱生成器")
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))

# 启用 LangSmith 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT


def check_config():
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_dashscope_api_key_here":
        raise ValueError("请在 .env 文件中填写 DASHSCOPE_API_KEY（通义千问 API Key）")
