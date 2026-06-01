import streamlit as st
from PIL import Image
from io import BytesIO
import re

st.set_page_config(
    page_title="私厨AI",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import LANGCHAIN_PROJECT, DASHSCOPE_API_KEY, QWEN_VISION_MODEL, QWEN_CHAT_MODEL
from src.vision import identify_ingredients
from src.agent import run_agent
from src.tools import generated_images, CHEF_TOOLS


def _run_agent_and_render(user_input: str):
    """运行 Agent，用 status 显示进度，完成后展示回复和图片。"""
    try:
        generated_images.clear()

        # 工具名 → 中文说明
        TOOL_LABELS = {
            "generate_recipe":    "📝 生成菜谱...",
            "get_nutrition_info": "🥗 查询营养信息...",
            "suggest_shopping":   "🛒 分析采购建议...",
            "create_dish_image":  "🎨 生成成品图片...",
        }

        reply = ""
        from src.agent import run_agent_with_steps
        with st.spinner("私厨小智思考中..."):
            reply, steps = run_agent_with_steps(user_input, st.session_state.messages)

        # 有工具调用才显示步骤
        if steps:
            with st.expander("🔧 调用了哪些工具", expanded=False):
                for step in steps:
                    label = TOOL_LABELS.get(step, f"🔧 {step}")
                    st.caption(f"✅ {label}")

        # 展示最终回复
        st.markdown(reply)

        # 展示生成的菜品图
        dish_images = {}
        for dish_name, img in generated_images.items():
            buf = BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            dish_images[dish_name] = img_bytes
            st.image(img_bytes, caption=f"🍽️ {dish_name} 成品效果图",
                     use_container_width=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "display_content": reply,
            "dish_images": dish_images,
        })
    except Exception as e:
        st.error(f"Agent 运行失败：{e}")

# ── 样式 ──────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #E8501A; margin-bottom: 0;
    }
    .ingredient-tag {
        display: inline-block;
        background: #FFF3EC; border: 1px solid #E8501A;
        color: #E8501A; border-radius: 20px;
        padding: 2px 10px; margin: 2px; font-size: 0.82rem;
    }
    .agent-step {
        background: #F0F7FF; border-left: 3px solid #4A90D9;
        padding: 6px 12px; margin: 4px 0;
        border-radius: 4px; font-size: 0.85rem; color: #333;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State 初始化 ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []
if "image_analyzed" not in st.session_state:
    st.session_state.image_analyzed = False


# ── 侧边栏 ────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">🍳 私厨AI</div>', unsafe_allow_html=True)
    st.caption("Agent 版 · 自主决策调用工具")
    st.divider()

    st.subheader("📸 上传食材图片")
    uploaded_file = st.file_uploader(
        "支持 JPG / PNG / WEBP",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        new_image = Image.open(BytesIO(uploaded_file.read()))
        img_key = f"{uploaded_file.name}_{new_image.size}"
        if st.session_state.get("img_key") != img_key:
            st.session_state.uploaded_image = new_image
            st.session_state.img_key = img_key
            st.session_state.image_analyzed = False
            st.session_state.ingredients = []
        st.image(st.session_state.uploaded_image, use_container_width=True)
    else:
        st.session_state.uploaded_image = None
        st.session_state.image_analyzed = False

    if st.session_state.ingredients:
        st.divider()
        st.caption("🥦 已识别食材")
        tags = "".join(
            f'<span class="ingredient-tag">{i}</span>'
            for i in st.session_state.ingredients
        )
        st.markdown(tags, unsafe_allow_html=True)

    st.divider()

    # Agent 工具箱说明
    st.caption("🔧 Agent 工具箱")
    for tool_name, desc in [
        ("generate_recipe",    "生成菜谱"),
        ("get_nutrition_info", "查询营养"),
        ("suggest_shopping",   "采购建议"),
        ("create_dish_image",  "生成成品图"),
        ("estimate_cooking_time", "估算烹饪时间"),
    ]:
        st.caption(f"&nbsp;&nbsp;• `{tool_name}` — {desc}")

    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ingredients = []
        st.session_state.image_analyzed = False
        st.session_state.uploaded_image = None
        generated_images.clear()
        st.rerun()

    st.divider()
    st.caption(f"📊 LangSmith：`{LANGCHAIN_PROJECT}`")
    if DASHSCOPE_API_KEY and DASHSCOPE_API_KEY != "your_dashscope_api_key_here":
        st.success("✅ 通义千问 已连接")
        st.caption(f"对话模型：`{QWEN_CHAT_MODEL}`")
        st.caption(f"视觉模型：`{QWEN_VISION_MODEL}`")
    else:
        st.error("⚠️ 请填写 DASHSCOPE_API_KEY")


# ── 主聊天区 ──────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🍳"):
        st.markdown("""你好！我是升级版**私厨小智** 👨‍🍳

我现在是一个 **AI Agent**，可以自主决定调用哪些工具来帮你：

- 📸 **上传图片** → 识别食材，自动推荐菜谱
- 🍽️ **问菜谱** → "用土豆和排骨能做什么？"
- 🥗 **问营养** → "番茄的营养价值是什么？"
- 🛒 **问采购** → "我只有鸡蛋，再买什么能多做几道菜？"
- 🖼️ **要图片** → "给我看看红烧肉的成品图"
- ⏱️ **问时间** → "番茄炒蛋需要多长时间？"

有什么需要尽管说！""")

# 渲染历史消息
for msg in st.session_state.messages:
    avatar = "🍳" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "user" and msg.get("has_image"):
            st.image(msg["image_preview"], width=200)
            st.markdown(msg.get("text_content", ""))
        else:
            content = msg.get("display_content") or msg.get("content", "")
            if content:
                st.markdown(content)
            # 恢复已生成的图片
            if msg.get("dish_images"):
                for dish_name, img_bytes in msg["dish_images"].items():
                    st.image(img_bytes, caption=f"🍽️ {dish_name} 成品效果图",
                             use_container_width=True)


# ── 输入框 ────────────────────────────────────────
user_input = st.chat_input("问我任何烹饪问题，或上传图片后发消息分析食材...")

if user_input:
    has_image = st.session_state.uploaded_image is not None
    image = st.session_state.uploaded_image

    # 显示用户消息
    with st.chat_message("user", avatar="🧑‍💻"):
        if has_image and not st.session_state.image_analyzed:
            st.image(image, width=200)
        st.markdown(user_input)

    # 构建发给 Agent 的输入
    agent_input = user_input

    if has_image and not st.session_state.image_analyzed:
        with st.chat_message("assistant", avatar="🍳"):
            with st.spinner("正在识别图片中的食材..."):
                try:
                    vision_result = identify_ingredients(image)
                    ingredients = vision_result["ingredients"]
                    states = vision_result.get("states", {})
                    scene = vision_result.get("scene", "")
                    st.session_state.ingredients = ingredients
                    st.session_state.image_analyzed = True

                    states_str = "、".join(
                        f"{k}({v})" for k, v in states.items()
                    ) if states else ""
                    agent_input = (
                        f"【图片食材识别结果】\n"
                        f"食材：{'、'.join(ingredients)}\n"
                        f"{'状态：' + states_str if states_str else ''}\n"
                        f"{'场景：' + scene if scene else ''}\n\n"
                        f"用户说：{user_input}"
                    )

                    img_buf = BytesIO()
                    img_copy = image.copy()
                    img_copy.thumbnail((300, 300), Image.LANCZOS)
                    if img_copy.mode not in ("RGB",):
                        img_copy = img_copy.convert("RGB")
                    img_copy.save(img_buf, format="JPEG")
                    img_preview = img_buf.getvalue()

                    st.session_state.messages.append({
                        "role": "user",
                        "content": agent_input,
                        "has_image": True,
                        "image_preview": img_preview,
                        "text_content": user_input,
                    })
                except Exception as e:
                    st.error(f"图片识别失败：{e}")
                    st.stop()

            # 运行 Agent
            with st.spinner("私厨小智正在思考并调用工具..."):
                _run_agent_and_render(agent_input)
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "display_content": user_input,
        })
        with st.chat_message("assistant", avatar="🍳"):
            with st.spinner("私厨小智正在思考并调用工具..."):
                _run_agent_and_render(user_input)


