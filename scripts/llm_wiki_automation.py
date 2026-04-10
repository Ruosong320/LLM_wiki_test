import os
import pathlib
import schedule
import time
import requests
from dotenv import load_dotenv

# -------------------------- 环境变量加载与全局配置 --------------------------
# 加载.env文件中的Qwen3.5配置
load_dotenv()
QWEN3_BASE_URL = os.getenv("LLM_URL")
QWEN3_MODEL = os.getenv("MODEL_NAME")
QWEN3_MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
QWEN3_TEMPERATURE = float(os.getenv("TEMPERATURE"))


# 配置LLM Wiki根目录（必修改：替换为你的PyCharm中LLM-Wiki-Vault的实际路径）
# 示例（Windows）：C:/Users/你的用户名/PyCharmProjects/LLM-Wiki-Vault
# 示例（Mac/Linux）：/Users/你的用户名/PyCharmProjects/LLM-Wiki-Vault
VAULT_PATH = "/Users/ruosongchen/PyCharmMiscProject/LLM_wiki_test"
RAW_PATH = os.path.join(VAULT_PATH, "raw")
WIKI_PATH = os.path.join(VAULT_PATH, "wiki")
SCHEMA_PATH = os.path.join(VAULT_PATH, "schema")
SCRIPTS_PATH = os.path.join(VAULT_PATH, "scripts")
LOG_PATH = os.path.join(SCHEMA_PATH, "lint-logs")

# 确保日志目录存在（巡检/摄入日志存放）
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


# -------------------------- 核心函数：调用本地Qwen3.5（适配27b-q4_K_M） --------------------------
def call_qwen3(prompt):
    """
    调用本地Qwen3.5:27b-q4_K_M，执行指定操作（摄入/查询/巡检）
    :param prompt: 唤起指令（包含schema规则、操作类型、细节）
    :return: Qwen3.5的响应结果（文本），失败则返回错误信息
    """
    # 构造请求头
    headers = {"Content-Type": "application/json"}
    # 构造请求参数（适配qwen3.5:27b-q4_K_M的推理特性）
    data = {
        "model": QWEN3_MODEL,
        "messages": [
            {"role": "system",
             "content": "你是LLM Wiki的专属维基管理员，严格遵循schema目录下的所有规则（structure.md、ingest.md、query.md、lint.md），执行操作时优先保证内容规范、引用完整，不编造信息、不修改raw目录文件。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": QWEN3_MAX_TOKENS,
        "temperature": QWEN3_TEMPERATURE,
        "stream": False
    }

    try:
        # 发送请求到本地Qwen3.5服务器
        response = requests.post(QWEN3_BASE_URL, json=data, headers=headers, timeout=300)  # 超时设为5分钟，适配长文本处理
        response.raise_for_status()  # 抛出HTTP错误（如404、500）
        result = response.json()
        # 提取Qwen3.5的响应内容（适配qwen3.5的返回格式）
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return f"Qwen3.5调用超时（超过5分钟），可能是处理长文本导致，建议检查模型推理速度或减少单次摄入的文件数量。"
    except Exception as e:
        return f"Qwen3.5调用失败：{str(e)}，请检查模型部署状态、接口地址和.env配置。"


# -------------------------- 功能函数1：资料摄入（核心，触发Wiki生成/更新） --------------------------
def ingest_raw_data():
    """
    自动检测raw目录新增/修改文件，调用Qwen3.5执行摄入，生成/更新Wiki页面
    流程：读取schema规则 → 检测raw目录变化 → 调用Qwen3.5解析 → 生成/更新Wiki → 保存摄入日志
    """
    # 1. 读取schema中的所有规则（确保Qwen3.5严格遵循）
    try:
        with open(os.path.join(SCHEMA_PATH, "ingest.md"), "r", encoding="utf-8") as f:
            ingest_rules = f.read()
        with open(os.path.join(SCHEMA_PATH, "structure.md"), "r", encoding="utf-8") as f:
            structure_rules = f.read()
        with open(os.path.join(SCHEMA_PATH, "template.md"), "r", encoding="utf-8") as f:
            template_rules = f.read()
    except Exception as e:
        log_content = f"摄入失败：读取schema规则失败，错误信息：{str(e)}，请检查schema目录下的文件是否完整。"
        print(log_content)
        # 保存错误日志
        log_file = os.path.join(LOG_PATH, f"{time.strftime('%Y%m%d_%H%M%S')}-ingest-error.md")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                f"# LLM Wiki 资料摄入错误日志\n## 摄入时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n## 错误信息：\n{log_content}")
        return

    # 2. 生成Qwen3.5的摄入指令（明确操作细节，适配qwen3.5的理解能力）
    prompt = f"""
请你作为LLM Wiki的维基管理员，严格遵循以下所有规则，执行资料摄入操作，适配qwen3.5:27b-q4_K_M模型的推理特性，避免生成冗余内容：
1.  结构与命名规则：{structure_rules}
2.  摄入规则：{ingest_rules}
3.  页面模板规则：{template_rules}

## 操作细节：
1.  检测目录：{RAW_PATH} 目录下所有新增、修改的文件（包括articles、papers、books、images子目录）；
2.  文件处理：优先读取Markdown格式文件，PDF格式若无法提取文字，直接标注“无法解析，需人工转换为Markdown”，不擅自处理；
3.  Wiki更新：按规则提取信息，生成/更新 {WIKI_PATH} 目录下的对应Wiki页面，生成后直接保存到对应子目录（如概念页放入wiki/concepts/）；
4.  格式要求：生成的Wiki页面编码为UTF-8，避免中文乱码，严格遵循template.md模板，引用来源标注完整；
5.  输出要求：详细输出摄入日志，包含“新增/更新的Wiki页面路径、引用的raw文件、操作结果（成功/失败）”，若有失败，标注具体原因。

## 注意事项（适配qwen3.5:27b-q4_K_M）：
- 解析raw文件时，优先提取核心信息，过滤冗余内容，避免生成过长文本，控制单页面字数在2000字以内；
- 若遇到多个raw文件涉及同一概念/实体，整合信息后更新Wiki页面，不重复生成；
- 若无法读取某文件，直接标注错误，不擅自创建文件或修改目录。
    """

    # 3. 调用Qwen3.5执行摄入操作
    print(f"开始执行资料摄入，时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    ingest_result = call_qwen3(prompt)

    # 4. 保存摄入日志（无论成功/失败，均记录）
    log_file = os.path.join(LOG_PATH, f"{time.strftime('%Y%m%d_%H%M%S')}-ingest-log.md")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            f"# LLM Wiki 资料摄入日志\n## 摄入时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n## 摄入结果：\n{ingest_result}")

    # 5. 终端输出结果，便于用户核对
    print(f"资料摄入完成，日志已保存至：{log_file}")
    print("摄入结果预览：\n", ingest_result[:500], "...（完整内容见日志）")


# -------------------------- 功能函数2：查询响应（用户交互，从Wiki提取内容） --------------------------
def query_wiki(query_content):
    """
    调用Qwen3.5，响应用户查询，优先从Wiki中提取内容，不直接检索raw目录
    :param query_content: 用户的查询内容（如「LLM的定义」「如何触发资料摄入」）
    :return: 查询响应结果（简洁、有条理，适配PyCharm终端显示）
    """
    # 1. 读取schema中的查询规则
    try:
        with open(os.path.join(SCHEMA_PATH, "query.md"), "r", encoding="utf-8") as f:
            query_rules = f.read()
    except Exception as e:
        return f"查询失败：读取查询规则失败，错误信息：{str(e)}，请检查schema/query.md文件是否存在。"

    # 2. 生成Qwen3.5的查询指令
    prompt = f"""
请你作为LLM Wiki的维基管理员，严格遵循以下查询规则，执行查询操作，适配PyCharm终端显示，内容简洁、有条理，优先用列表呈现：
查询规则：{query_rules}

## 操作细节：
1.  查询内容：{query_content}
2.  检索范围：优先从 {WIKI_PATH} 目录中检索对应Wiki页面，整合核心内容，不直接检索raw目录；
3.  响应要求：标注引用的Wiki页面路径（如 wiki/concepts/LLM.md），若Wiki中无相关内容，提示用户导入raw资料触发摄入；
4.  格式要求：避免过长段落，关键信息用列表呈现，不冗余，适配qwen3.5:27b-q4_K_M的输出特性。
    """

    # 3. 调用Qwen3.5执行查询
    print(f"正在查询：{query_content}，请等待...")
    query_result = call_qwen3(prompt)
    return query_result


# -------------------------- 功能函数3：定期巡检（维护Wiki准确性，自动修复简单问题） --------------------------
def lint_wiki():
    """
    调用Qwen3.5，全量巡检Wiki和raw目录，检查格式、内容、目录问题，生成报告并修复简单问题
    流程：读取schema规则 → 全量巡检 → 自动修复简单问题 → 生成巡检报告 → 保存日志
    """
    # 1. 读取schema中的巡检规则和结构规则
    try:
        with open(os.path.join(SCHEMA_PATH, "lint.md"), "r", encoding="utf-8") as f:
            lint_rules = f.read()
        with open(os.path.join(SCHEMA_PATH, "structure.md"), "r", encoding="utf-8") as f:
            structure_rules = f.read()
    except Exception as e:
        log_content = f"巡检失败：读取schema规则失败，错误信息：{str(e)}，请检查schema目录下的文件是否完整。"
        print(log_content)
        # 保存错误日志
        log_file = os.path.join(LOG_PATH, f"{time.strftime('%Y%m%d_%H%M%S')}-lint-error.md")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                f"# LLM Wiki 巡检错误日志\n## 巡检时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n## 错误信息：\n{log_content}")
        return

    # 2. 生成Qwen3.5的巡检指令（适配qwen3.5:27b-q4_K_M的推理能力）
    prompt = f"""
请你作为LLM Wiki的维基管理员，严格遵循以下所有规则，执行全量巡检操作，适配qwen3.5:27b-q4_K_M模型，细致检查每一个页面：
1.  结构与命名规则：{structure_rules}
2.  巡检规则：{lint_rules}

## 操作细节：
1.  巡检范围：{WIKI_PATH} 目录下所有Wiki页面、{RAW_PATH} 目录下所有原始资料；
2.  检查重点：按“格式问题→内容问题→目录问题”的优先级，细致检查，不遗漏任何小问题；
3.  问题处理：简单问题（命名不规范、失效链接）自动修复，修复后标注“修复内容”；复杂问题（内容矛盾、过时）无法修复，标注“待人工处理”；
4.  输出要求：生成完整的巡检报告，包含“问题分类、涉及页面、修复结果、待处理事项”，格式清晰，适配PyCharm终端显示和Markdown预览；
5.  日志留存：修复后的Wiki页面直接同步到 {WIKI_PATH} 目录，巡检报告后续由脚本保存到 {LOG_PATH}。

## 注意事项（适配qwen3.5:27b-q4_K_M）：
- 巡检时仅读取目录和文件，不修改raw目录任何文件，不擅自创建目录；
- 修复Wiki页面时，不覆盖人工编辑的内容，仅修复格式和简单错误；
- 若无法读取某页面，标注错误，不擅自删除或修改页面。
    """

    # 3. 调用Qwen3.5执行巡检操作
    print(f"开始执行全量巡检，时间：{time.strftime('%Y-%m-%d %H:%M:%S')}，请耐心等待...")
    lint_result = call_qwen3(prompt)

    # 4. 保存巡检日志和待处理事项
    # 保存巡检报告
    log_file = os.path.join(LOG_PATH, f"{time.strftime('%Y%m%d_%H%M%S')}-lint-log.md")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"# LLM Wiki 巡检日志\n## 巡检时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n## 巡检结果：\n{lint_result}")

    # 生成待处理事项笔记（若有）
    if "待人工处理" in lint_result:
        pending_file = os.path.join(SCHEMA_PATH, "lint-pending.md")
        with open(pending_file, "w", encoding="utf-8") as f:
            f.write(f"# LLM Wiki 巡检待处理事项\n## 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n## 待处理内容：\n")
            # 提取待处理事项，写入文件
            pending_items = [line for line in lint_result.split("\n") if "待人工处理" in line]
            for item in pending_items:
                f.write(f"- {item}\n")
        print(f"发现待处理事项，已保存至：{pending_file}")

    # 5. 终端输出结果
    print(f"巡检完成，日志已保存至：{log_file}")
    print("巡检结果预览：\n", lint_result[:500], "...（完整内容见日志）")


# -------------------------- 功能函数4：手动触发操作（适配用户手动控制） --------------------------
def manual_trigger(action_type, detail=""):
    """
    手动触发操作（摄入/查询/巡检），适配用户在PyCharm终端手动输入指令
    :param action_type: 操作类型（ingest/query/lint）
    :param detail: 操作细节（如query的查询内容，ingest无需细节）
    """
    if action_type.lower() == "ingest":
        print("手动触发资料摄入...")
        ingest_raw_data()
    elif action_type.lower() == "query":
        if not detail:
            print("查询失败：请输入查询内容（如 'query LLM的定义'）")
            return
        result = query_wiki(detail)
        print(f"\n查询结果：\n{result}")
    elif action_type.lower() == "lint":
        print("手动触发全量巡检...")
        lint_wiki()
    else:
        print("无效操作类型，仅支持：ingest（资料摄入）、query（查询）、lint（巡检）")


# -------------------------- 定时任务：自动触发摄入和巡检（无人干预） --------------------------
def schedule_tasks():
    """
    配置定时任务，实现无人干预的自动化（适配qwen3.5:27b-q4_K_M的推理速度）
    可按需修改定时时间，避免同时执行多个操作（导致显存溢出）
    """
    # 每天20:00，自动检测raw目录新增文件，执行摄入操作（避开巡检时间）
    schedule.every().day.at("20:00").do(ingest_raw_data)

    # 每天22:00，自动执行全量巡检（摄入完成后，避免资源冲突）
    schedule.every().day.at("22:00").do(lint_wiki)

    # 持续运行定时任务，每分钟检查一次
    print("定时任务已启动，开始自动执行Qwen3.5操作（摄入：20:00，巡检：22:00）...")
    print("若需手动触发操作，按 Ctrl+C 停止定时任务，输入对应指令即可。")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次任务
    except KeyboardInterrupt:
        print("\n定时任务已停止，可手动触发操作。")


# -------------------------- 执行入口（用户交互，可手动/自动触发） --------------------------
if __name__ == "__main__":
    # 启动提示（适配用户操作）
    print("=" * 50)
    print("LLM Wiki 自动化脚本（适配qwen3.5:27b-q4_K_M）")
    print("=" * 50)
    print("操作说明：")
    print("1. 输入 'ingest' → 手动触发资料摄入")
    print("2. 输入 'query 内容' → 手动触发查询（如 'query LLM的定义'）")
    print("3. 输入 'lint' → 手动触发全量巡检")
    print("4. 输入 'auto' → 启动定时任务（自动摄入+巡检）")
    print("5. 输入 'quit' → 退出脚本")
    print("=" * 50)

    # 交互循环
    while True:
        user_input = input("请输入操作指令：")
        if user_input.lower() == "quit":
            print("退出脚本，感谢使用！")
            break
        elif user_input.lower() == "ingest":
            manual_trigger("ingest")
        elif user_input.lower() == "lint":
            manual_trigger("lint")
        elif user_input.lower() == "auto":
            schedule_tasks()
        elif user_input.lower().startswith("query"):
            # 提取查询内容（如 "query LLM的定义" → 提取 "LLM的定义"）
            query_detail = user_input[6:].strip()
            if not query_detail:
                print("查询失败：请输入查询内容（如 'query LLM的定义'）")
                continue
            manual_trigger("query", query_detail)
        else:
            print("无效指令，请重新输入（参考操作说明）。")