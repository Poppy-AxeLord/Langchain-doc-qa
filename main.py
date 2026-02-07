# 主程序入口
import os
import traceback

# 导入各个模块
from config.settings import CONFIG
from cache.qa_cache import init_qa_cache
from cache.pdf_cache import get_pdf_file_hash, load_pdf_topic_from_cache, save_cache_pdf_topic
from retrieval.pdf_loader import load_and_split_pdf, get_pdf_basic_info
from agents.agent_builder import build_qa_agent
from utils.file_utils import ensure_directory

# 导入模型相关
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ------------------------------ 模型初始化 ------------------------------
# 1. 嵌入模型初始化（用于文档向量化，适配Chroma向量库）
embeddings = HuggingFaceEmbeddings(
    model_name=CONFIG["embedding_path"],
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 2. LLM模型初始化（qwen-plus，核心问答与推理模型）
llm = ChatOpenAI(
    model=CONFIG["llm_model"],
    temperature=CONFIG["llm_temperature"],
    api_key=str(CONFIG["qwen_api_key"]),  # 唯一核心：确保API Key是字符串
    base_url=CONFIG["qwen_base_url"],
    cache=True,
    timeout=10,  # 新增：10秒超时，避免卡死
)

# 3. 重排序模型初始化（用于检索结果精排，提升相关性）
reranker = HuggingFaceCrossEncoder(
    model_name=CONFIG["rerank_path"],
    model_kwargs={"device": "cpu"}
)

# ------------------------------ 初始化函数 ------------------------------
def initialize_qa_system():
    """
    初始化问答系统
    :return: 问答函数、清空历史函数、清空缓存函数
    """
    # 初始化缓存
    init_qa_cache()
    
    # 1. 加载PDF并初始化向量库
    docs = load_and_split_pdf()
    # 初始化Chroma向量库，存储文档嵌入
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CONFIG["chroma_dir"]
    )
    # 创建向量检索器
    retriever = vector_db.as_retriever(search_kwargs={"k": CONFIG["retrieve_top_k"]})

    # 加载/提取PDF基础信息（优先从缓存加载）
    pdf_hash = get_pdf_file_hash(CONFIG["pdf_path"])
    pdf_info = load_pdf_topic_from_cache(pdf_hash)
    if not pdf_info:
        pdf_info = get_pdf_basic_info(docs, llm)
        save_cache_pdf_topic(pdf_hash, pdf_info)

    # 构建问答Agent
    qa_func, clear_history_func, clear_all_cache_func = build_qa_agent(
        llm=llm,
        docs=docs,
        embeddings=embeddings,
        vector_db=vector_db,
        retriever=retriever,
        pdf_hash=pdf_hash,
        pdf_info=pdf_info,
        reranker=reranker
    )

    return qa_func, clear_history_func, clear_all_cache_func

# ------------------------------ 主程序入口（程序启动入口） ------------------------------
if __name__ == "__main__":
    # 创建必要目录（缓存目录+向量库目录）
    ensure_directory("./cache")
    ensure_directory(CONFIG["chroma_dir"])
    
    try:
        # 初始化问答助手
        print("🚀 初始化PDF问答助手（分层记忆+自定义缓存+工具调用版）...")
        qa_func, clear_history_func, clear_all_cache_func = initialize_qa_system()
        print("✅ 助手就绪！")
        # 打印命令说明
        print("📖 命令说明：")
        print("  - quit：退出程序")
        print("  - clear：清空对话记忆")
        print("  - clear_cache：清空所有缓存")
        print("  - 支持工具调用：查主题/关键词、查具体内容、按页码查内容")
        print("-" * 60)

        # 循环接收用户输入
        while True:
            question = input("\n请输入你的问题：")
            
            # 命令处理
            if question.lower() == "quit":
                print("\n👋 再见！")
                break
            elif question.lower() == "clear":
                print(f"\n{clear_history_func()}")
                continue
            elif question.lower() == "clear_cache":
                print(f"\n{clear_all_cache_func()}")
                continue
            
            # 执行问答并输出结果
            answer, sources = qa_func(question)
            print(f"\n📝 回答：{answer}")
            if sources:
                print(f"📎 来源：{', '.join(sources)}")
            print("-" * 60)

    # 异常处理
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出！")
    except FileNotFoundError as e:
        print(f"\n❌ 错误：{e}，请确保PDF文件路径正确")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")
        traceback.print_exc()
