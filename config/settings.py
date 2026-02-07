import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 全局配置
CONFIG = {
    "qwen_api_key": os.getenv("OPENAI_API_KEY"),
    "qwen_base_url": os.getenv("OPENAI_API_BASE_URL"),
    "llm_model": os.getenv("OPENAI_MODEL"),
    "pdf_path": "./assets/file.pdf",
    "embedding_path": "./models/bge-small-zh-v1.5",
    "rerank_path": "./models/bge-reranker-base",
    "chroma_dir": "./chroma_db",
    "llm_temperature": 0.1,
    "retrieve_top_k": 2,
    "max_chunk_size": 300, # 从500缩小到300，更精准切分
    "min_chunk_size": 100,
    "chunk_overlap": 30, # 重叠部分相应减少
    "short_term_max_rounds": 3 # 短期记忆最大轮数
}
