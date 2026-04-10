import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
base_url = os.getenv("LLM_URL")
model = os.getenv("MODEL_NAME")
max_tokens = int(os.getenv("MAX_TOKENS"))
temperature = float(os.getenv("TEMPERATURE"))


# 测试Qwen3.5长文本响应（模拟Wiki页面生成场景）
def test_qwen3_long_text():
    # 构造请求头（有API Key则添加）
    headers = {"Content-Type": "application/json"}

    # 模拟资料摄入时的指令（测试模型解析能力）
    test_prompt = """
请你作为LLM Wiki的维基管理员，解析以下简单的raw资料，按schema/template.md的概念页模板，生成一个简短的Wiki页面：
raw资料：LLM（大语言模型）是指参数量巨大、能理解和生成人类语言的人工智能模型，核心特征是具备上下文理解、逻辑推理和内容生成能力，常见应用场景包括知识管理、内容创作等。
要求：生成的Wiki页面命名为LLM.md，放入wiki/concepts/目录，包含定义、核心特征、应用场景，标注引用来源为raw/articles/LLM基础介绍-20260410.md。
    """

    data = {
        "model": model,
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False  # 关闭流式输出，便于脚本后续处理结果
    }

    try:
        response = requests.post(base_url, json=data, headers=headers)
        response.raise_for_status()  # 抛出HTTP错误（如接口不可用）
        result = response.json()
        print("Qwen3.5长文本响应成功，生成的Wiki内容：\n", result["choices"][0]["message"]["content"])
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print("Qwen3.5调用失败（长文本测试）：", str(e))
        # 新增：若出现invalid link相关错误，直接提示排查接口地址
        if "invalid link" in str(e).lower() or "connection" in str(e).lower():
            print("重点排查：Qwen3.5接口地址（QWEN3_BASE_URL）无效或无法连接，请核对.env文件中的地址和模型部署状态")
        return None


if __name__ == "__main__":
    test_qwen3_long_text()