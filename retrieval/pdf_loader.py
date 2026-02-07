import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_index.core import SimpleDirectoryReader

from config.settings import CONFIG

def semantic_split(docs, max_chunk_len=300, min_chunk_len=100):
    """
    语义分块函数
    :param docs: 原始 Document 列表（你的 raw_docs）
    :param max_chunk_len: 单个块最大字符数
    :param min_chunk_len: 单个块最小字符数
    :return: 语义完整的 Document 分块列表
    """
    final_split_docs = []

    for raw_doc in docs:
        raw_text = raw_doc.page_content.strip()
        doc_metadata = raw_doc.metadata  # 保留原始页码等元数据

        # 步骤1：优先按段落（空行）拆分
        paragraph_chunks = re.split(r'\n\n+', raw_text)

        for para in paragraph_chunks:
            para = para.strip()
            if not para:
                continue

            # 步骤2：段落尺寸合适，直接封装为 Document
            if min_chunk_len <= len(para) <= max_chunk_len:
                para_doc = Document(page_content=para, metadata=doc_metadata)
                final_split_docs.append(para_doc)
                continue

            # 步骤3：段落过长，按中文句子拆分（保留标点）
            if len(para) > max_chunk_len:
                sent_parts = re.split(r'([。！？；])', para)
                temp_sent = ""
                for part in sent_parts:
                    temp_sent += part
                    # 满足尺寸要求时，封装为 Document
                    if (len(temp_sent) >= min_chunk_len and part in ["。", "！", "？", "；"]) or len(temp_sent) >= max_chunk_len:
                        if temp_sent.strip():
                            sent_doc = Document(page_content=temp_sent.strip(), metadata=doc_metadata)
                            final_split_docs.append(sent_doc)
                        temp_sent = ""
                # 处理最后一个剩余句子
                if temp_sent.strip():
                    last_sent_doc = Document(page_content=temp_sent.strip(), metadata=doc_metadata)
                    final_split_docs.append(last_sent_doc)

    return final_split_docs

def load_and_split_pdf():
    """
    加载PDF并执行语义分块
    :return: 分块后的文档列表
    """
    # 校验PDF文件是否存在
    if not os.path.exists(CONFIG["pdf_path"]):
        raise FileNotFoundError(f"PDF文件不存在：{CONFIG['pdf_path']}")
    
    # 加载PDF文档
    llama_docs = SimpleDirectoryReader(input_files=[CONFIG["pdf_path"]]).load_data()
    # 转换为LangChain Document格式
    raw_docs = [
        Document(page_content=doc.text.strip(), metadata={"page": doc.metadata.get("page_label", "未知")})
        for doc in llama_docs
    ]
    print(f"📄 PDF加载完成，共{len(raw_docs)}页")

    # 执行语义分块
    split_docs = semantic_split(
        docs=raw_docs,
        max_chunk_len=CONFIG["max_chunk_size"],
        min_chunk_len=CONFIG["min_chunk_size"]
    )
    print(f"✂️ PDF分块完成，共{len(split_docs)}个片段")

    return split_docs

def get_pdf_basic_info(docs, llm):
    """
    提取PDF基础信息（核心关键词+主题，用于相关性判断）
    :param docs: 文档列表
    :param llm: 语言模型
    :return: PDF基础信息字典
    """
    # 提取前5个文档的前200字作为核心内容
    core_content = "\n".join([doc.page_content[:200] for doc in docs[:5]])
    
    # 构建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """请总结以下内容的核心信息：
        1. 输出5个核心关键词（用逗号分隔）
        2. 输出1句话总结核心主题（不超过80字）
        输出格式：先写关键词，换行后写主题"""),
        ("human", f"内容：{core_content}")
    ])
    
    # 构建信息提取链
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({}).strip()
        lines = result.split("\n")
        
        # 解析关键词和主题
        keywords = lines[0].split(",") if len(lines)>=1 else ["未知"]
        topic = lines[1].strip() if len(lines)>=2 else "未知PDF文档"
        
        # 清洗关键词并补全5个
        keywords = [k.strip() for k in keywords if k.strip()]
        if len(keywords) < 5:
            keywords += ["未知"] * (5 - len(keywords))
        
        return {"keywords": keywords[:5], "topic": topic}
    except Exception as e:
        print(f"⚠️ 提取PDF信息失败：{e}")
        return {"keywords": ["未知"]*5, "topic": "未知PDF文档"}
