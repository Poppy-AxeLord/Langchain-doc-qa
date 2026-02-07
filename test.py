import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # 加载OpenAI兼容模型（qwen-plus）

# 加载.env文件中的环境变量
load_dotenv()

# 1. 加载你的环境配置（和你提供的变量对应）
qwen_api_key = os.getenv("OPENAI_API_KEY")
qwen_base_url = os.getenv("OPENAI_API_BASE_URL")
llm_model = os.getenv("OPENAI_MODEL")

# 2. 初始化通义千问 LLM 实例
try:
    llm = ChatOpenAI(
        api_key=qwen_api_key,
        base_url=qwen_base_url,
        model=llm_model,
    )

    # 3. 构建简单的测试提示词
    prompt = PromptTemplate(
        input_variables=["question"],
        template="请简洁回答这个问题：{question}"
    )

    # 4. 拼接提示词并调用 LLM
    chain = prompt | llm
    response = chain.invoke({"question": "LangChain 连接通义千问是否正常？"})

    # 5. 输出结果（验证交互是否成功）
    print("=" * 50)
    print("LLM 响应结果：")
    print(response)
    print("=" * 50)
    print("✅ 恭喜！LLM 交互正常！")

except Exception as e:
    print("=" * 50)
    print(e)
    print("❌ LLM 交互失败，错误信息：")
