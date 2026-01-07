# 1. Python 内置模块
import os                   # 用于文件路径操作、目录创建
import re                   # 用于正则表达式匹配（文本处理、分词过滤）
import json                 # 用于数据序列化（缓存相关）
import time                 # 用于计时（辅助异步性能监控）
import asyncio              # 用于异步并行检索（提升检索效率）
import traceback            # 用于异常堆栈打印（错误排查）

# 2. 第三方工具模块
import jieba                # 用于中文分词（适配BM25关键词检索）
from dotenv import load_dotenv  # 用于加载.env文件中的环境变量（LLM密钥等）

# 3. 类型注解与数据验证模块
from typing import List, Dict, Optional  # 用于类型提示，提升代码可读性
from pydantic import BaseModel, Field    # 用于数据模型验证（未直接使用，保留兼容）

# 4. LangChain 核心模块
from langchain_core.documents import Document  # 定义文档数据结构
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate  # 定义提示词模板
from langchain_core.output_parsers import StrOutputParser  # LLM输出文本解析器
from langchain_core.caches import InMemoryCache  # LLM原生内存缓存
from langchain_core.globals import set_llm_cache  # 配置LLM全局缓存
from langchain_core.messages import HumanMessage, AIMessage  # 定义对话消息类型
from langchain_core.tools import tool  # 用于定义Agent可调用工具

# 5. LangChain 模型与存储模块
from langchain_openai import ChatOpenAI  # 加载OpenAI兼容模型（qwen-plus）
from langchain_huggingface import HuggingFaceEmbeddings  # 加载HuggingFace嵌入模型
from langchain_chroma import Chroma  # 加载Chroma向量数据库（文档存储与检索）
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分块器（备用，实际使用语义分块）
from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # 加载交叉编码器（文档重排序）

# 6. LangChain Agent 模块
from langchain.agents import create_agent  # 创建工具调用Agent

# 7. 文档加载与BM25检索模块
from llama_index.core import SimpleDirectoryReader  # 加载PDF文档
from rank_bm25 import BM25Okapi  # 加载BM25关键词检索器

# 8. 自定义模块
from cache_utils import get_pdf_file_hash, load_pdf_topic_from_cache, save_cache_pdf_topic  # PDF缓存工具
from layered_memory import LayeredMemoryManager  # 分层记忆管理器（短期+长期记忆）

# ------------------------------ 全局配置 ------------------------------
load_dotenv()
CONFIG = {
    "qwen_api_key": os.getenv("QWEN_API_KEY"),
    "qwen_base_url": os.getenv("QWEN_BASE_URL"),
    "pdf_path": "./assets/file.pdf",
    "embedding_path": "./models/bge-small-zh-v1.5",
    "rerank_path": "./models/bge-reranker-base",
    "chroma_dir": "./chroma_db",
    "llm_model": "qwen-plus",
    "llm_temperature": 0.1,
    "retrieve_top_k": 2,
    "max_chunk_size": 300, # 从500缩小到300，更精准切分
    "min_chunk_size": 100,
    "chunk_overlap": 30, # 重叠部分相应减少
    "short_term_max_rounds": 3 # 短期记忆最大轮数
}

# ------------------------------ 缓存配置 ------------------------------
# 1. LLM原生内存缓存（提升重复查询效率）
llm_cache = InMemoryCache()
set_llm_cache(llm_cache)

# 2. 自定义业务级缓存
qa_cache = {}          # 缓存最终问答结果：key=标准化问题，value=(回答, 来源)
relevance_cache = {}   # 缓存相关性判断：key=标准化问题，value=(是否相关, 理由)

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
    api_key=CONFIG["qwen_api_key"],
    base_url=CONFIG["qwen_base_url"],
    cache=True  # 开启LLM原生缓存兜底
)

# 3. 重排序模型初始化（用于检索结果精排，提升相关性）
reranker = HuggingFaceCrossEncoder(
    model_name=CONFIG["rerank_path"],
    model_kwargs={"device": "cpu"}
)

# ------------------------------ 核心功能函数 ------------------------------
# 标准化问题（统一格式，确保相同问题命中缓存）
def normalize_question(question):
    return question.strip().lower()

# 多查询生成核心函数（生成同义查询，提升检索召回率）
def generate_multi_queries(original_query, llm):
    multi_query_prompt = PromptTemplate(
        template="""你是查询优化助手，为原始问题生成3个同义/相关查询，用于提升检索召回率。
        规则：
        1.  仅返回3个查询，每行1个，无编号、无额外解释；
        2.  语义一致，仅调整表述（替换同义词/专业术语）；
        3.  适配金融财报场景，避免口语化；
        4.  不重复表述，不添加多余内容。

        原始问题：{original_query}""",
        input_variables=["original_query"]
    )

    # 构建多查询生成链（LangChain 1.0+ Runnable 写法）
    multi_query_chain = multi_query_prompt | llm | StrOutputParser()

    try:
        generated_queries_text = multi_query_chain.invoke({"original_query": original_query})
        generated_queries = [q.strip() for q in generated_queries_text.split("\n") if q.strip()]
        all_queries = list(set(generated_queries + [original_query]))  # 去重+合并原始查询
        final_queries = [q for q in all_queries if q] or [original_query]  # 过滤空值
        return final_queries[:4]  # 限制最多4个查询
    except Exception as e:
        print(f"⚠️ 多查询生成异常：{e}，已降级为原始查询")
        return [original_query]

# 语义分块函数（按段落+句子拆分，保证文档语义完整性）
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

# 加载PDF并执行语义分块
def load_and_split_pdf():
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

# 提取PDF基础信息（核心关键词+主题，用于相关性判断）
def get_pdf_basic_info(docs):
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

# 构建问答链（核心：整合RAG+Agent+分层记忆+工具调用）
def build_qa_chain():
    # 1. 加载PDF并初始化向量库
    docs = load_and_split_pdf()
    split_docs = docs  # 关键：获取split_docs，解决作用域问题
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
        pdf_info = get_pdf_basic_info(docs)
        save_cache_pdf_topic(pdf_hash, pdf_info)

    # ========== BM25 关键词检索器初始化（解决向量检索对关键词不敏感问题） ==========
    # 1. 提取所有文档片段的文本和元数据
    doc_texts = [doc.page_content for doc in split_docs]
    doc_metadata = [doc.metadata for doc in split_docs]

    # 2. 中文分词函数（适配BM25检索）
    def chinese_tokenizer(text):
        # 过滤特殊字符，保留有效文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\@\.]', ' ', text)
        # 通用停用词表
        stop_words = {"的", "了", "是", "在", "和", "有", "我", "你", "他", "这个", "那个"}
        # 分词并过滤停用词
        tokens = [
            token.strip()
            for token in jieba.cut(text)
            if token.strip() not in stop_words
        ]
        return tokens

    # 3. 对文档集进行分词，初始化BM25
    tokenized_corpus = [chinese_tokenizer(text) for text in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. BM25 关键词检索函数（带核心词加权，提升金融术语匹配度）
    def bm25_retrieve(query, top_k=2):
        # 对查询进行分词
        tokenized_query = chinese_tokenizer(query)
        # 金融财报核心词（加权提升匹配优先级）
        core_words = {"营收", "增速", "同比", "净利润", "利润", "数据"}
        
        # 计算每个文档的得分（核心词权重×2，普通词权重×1）
        scores = []
        for doc_tokens in tokenized_corpus:
            score = 0
            for token in tokenized_query:
                if token in core_words:
                    score += bm25.idf.get(token, 0) * (doc_tokens.count(token) / len(doc_tokens)) * 2
                else:
                    score += bm25.idf.get(token, 0) * (doc_tokens.count(token) / len(doc_tokens)) * 1
            scores.append(score)
        
        # 按得分排序，取Top-K文档
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_docs = [
            Document(page_content=doc_texts[idx], metadata=doc_metadata[idx])
            for idx in top_indices
        ]
        return top_docs

    # ========== 异步并行检索辅助函数（抵消多查询带来的耗时增加） ==========
    # 单个查询的异步检索：并行执行 BM25 和 向量检索
    async def async_retrieve_single_query(single_query, bm25_retrieve, retriever, top_k):
        # 使用 asyncio.to_thread 包装同步函数，实现并行执行
        bm25_docs_task = asyncio.to_thread(bm25_retrieve, single_query, top_k)
        vector_docs_task = asyncio.to_thread(retriever.invoke, single_query)
        # 并行等待两个任务完成，获取结果
        bm25_docs, vector_docs = await asyncio.gather(bm25_docs_task, vector_docs_task)
        return bm25_docs + vector_docs

    # 多查询异步检索主函数：生成多查询→并行检索→合并去重
    async def async_multi_query_retrieve(original_query, llm, bm25_retrieve, retriever, top_k=2):
        # 步骤1：生成多查询
        multi_queries = generate_multi_queries(original_query, llm)
        print(f"\n🔍 生成多查询列表：{multi_queries}（共{len(multi_queries)}个）")

        # 步骤2：创建所有查询的异步检索任务
        tasks = []
        for single_query in multi_queries:
            task = asyncio.create_task(async_retrieve_single_query(single_query, bm25_retrieve, retriever, top_k))
            tasks.append(task)

        # 步骤3：并行等待所有任务完成
        all_query_results = await asyncio.gather(*tasks)

        # 步骤4：合并所有检索结果并去重
        all_retrieved_docs = []
        doc_content_map = {}  # 按内容去重，避免重复文档
        for query_docs in all_query_results:
            for doc in query_docs:
                doc_content = doc.page_content.strip()
                if doc_content not in doc_content_map:
                    doc_content_map[doc_content] = doc
                    all_retrieved_docs.append(doc)

        return all_retrieved_docs

    # 同步包装函数（适配现有同步代码架构，无需重构其他逻辑）
    def multi_query_parallel_retrieve(original_query, llm, bm25_retrieve, retriever, top_k=2):
        return asyncio.run(async_multi_query_retrieve(original_query, llm, bm25_retrieve, retriever, top_k))

    # 文档重排序函数（对检索结果精排，提升相关性排序准确性）
    def rerank_docs(query, retrieved_docs):
        """
        对混合检索后的文档进行精细语义排序
        :param query: 用户查询
        :param retrieved_docs: 混合检索得到的 Document 列表
        :return: 重排序后的 Document 列表（异常时返回原列表）
        """
        # 强兜底：模型未初始化/无查询/无候选文档，直接返回原列表，不影响原有功能
        if not reranker or not query.strip() or not retrieved_docs:
            return retrieved_docs

        try:
            # 构造 (query, doc_content) 配对列表，适配交叉编码器输入
            query_doc_pairs = [(query, doc.page_content) for doc in retrieved_docs]
            # 计算相关性得分
            scores = reranker.score(query_doc_pairs)

            # 打印得分，方便调试
            print(f"\n===== Rerank 相关性得分 =====")
            for idx, score in enumerate(scores):
                print(f"文档{idx+1} 相关性得分：{score}")
                
            # 绑定文档与得分，按得分降序排序
            doc_score_pairs = list(zip(retrieved_docs, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            # 返回排序后的文档
            return [pair[0] for pair in doc_score_pairs]
        except Exception as e:
            # 异常时兜底返回原始文档，避免检索流程中断
            print(f"⚠️ Rerank 执行异常，返回原始检索结果：{e}")
            return retrieved_docs

    # ========== 混合检索函数（BM25+Chroma+多查询并行+Rerank，核心检索逻辑） ==========
    def hybrid_retrieve(query: str, top_k: int = 2) -> list[Document]:
        """
        混合检索函数（BM25关键词检索 + Chroma向量检索）
        核心功能：融合关键词匹配的精准性和语义匹配的泛化性，提升检索召回率和精准度
        :param query: 用户查询语句
        :param top_k: 单检索策略返回的顶部片段数量
        :return: 合并去重后的高质量文档片段列表
        """
        # 参数合法性校验
        if not isinstance(query, str) or len(query.strip()) == 0:
            return []
        if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            top_k = 2

        # 2. 多查询异步并行检索（核心修改点：提升召回率的同时保证性能）
        all_retrieved_docs = multi_query_parallel_retrieve(
            original_query=query,
            llm=llm,
            bm25_retrieve=bm25_retrieve,
            retriever=retriever,
            top_k=top_k
        )

        # 3. 结果去重（基于文档内容去重，避免重复片段干扰后续LLM推理）
        doc_content_unique_map = {}  # key: 文档内容（去重标识）, value: Document对象
        for doc in all_retrieved_docs:
            doc_content = doc.page_content.strip()
            if doc_content not in doc_content_unique_map:
                doc_content_unique_map[doc_content] = doc

        # 4. 结果裁剪（控制返回数量，避免过多片段增加LLM推理成本）
        final_retrieved_docs = list(doc_content_unique_map.values())[:top_k * 2]  # 取2倍top_k，兼顾召回率和性能
        # 5. 文档重排序（提升检索结果相关性）
        final_retrieved_docs = rerank_docs(query, final_retrieved_docs)

        return final_retrieved_docs

    # 3. 初始化分层记忆（短期记忆+长期记忆，提升对话连贯性）
    memory_manager = LayeredMemoryManager(
        llm=llm,
        short_term_max_rounds=CONFIG["short_term_max_rounds"]
    )

    # 5. 构建降级用回答链（Agent调用失败时，使用基础RAG兜底）
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业的文档问答助手，仅基于提供的文档内容回答问题：
        1. 严格基于文档内容，不编造信息
        2. 结合历史对话的核心信息和最新上下文理解问题
        3. 回答简洁明了，直击要点"""),
        ("human", """历史对话核心信息：{memory}
        文档内容：{context}
        问题：{question}""")
    ])

    # 构建基础回答链
    answer_chain = answer_prompt | llm | StrOutputParser()

    # ========== Agent 工具定义（3个核心工具，覆盖PDF查询核心场景） ==========
    @tool
    def pdf_meta_query_tool() -> str:
        """
        专门用于查询当前PDF的核心元信息（主题+5个关键词）。
        当用户提问类似「这份PDF讲了什么？」「PDF关键词有哪些？」「文档主题是什么？」时调用。
        输出返回PDF主题和关键词的原始信息，不添加任何额外话术、寒暄、引导提问等无关内容。
        """
        meta_info = f"""
        PDF核心元信息
        - 核心主题：{pdf_info['topic']}
        - 核心关键词：{', '.join(pdf_info['keywords'])}
        """
        return meta_info.strip()

    @tool
    def pdf_vector_retrieve_tool(query: str, top_k: int = 2) -> str:
        """
        专门用于检索PDF中与用户问题相关的精准文本片段，用于回答具体概念、数据、方法等非主题类问题。
        触发条件：当用户提问不涉及PDF主题/关键词，也不涉及页码时调用。
        参数说明：
        - query：必填项，传入用户完整查询问题。
        - top_k：可选项，检索返回的文本片段数量，默认2。
        """
        try:
            # 简单参数校验
            if not query.strip():
                return "错误：查询内容不能为空！"
            top_k = max(1, min(top_k, 5))  # 限制top_k范围，避免无效检索
            
            # 改用混合检索（核心：提升检索效果）
            # 原代码：retrieved_docs = retriever.invoke(query)
            retrieved_docs = hybrid_retrieve(query, top_k=top_k)
            
            # 格式化检索结果，附带页码信息
            context_str = ""
            for idx, doc in enumerate(retrieved_docs, 1):
                page_num = doc.metadata.get("page", "未知")
                context_str += f"【检索片段{idx}（页码：{page_num}）】\n{doc.page_content}\n\n"
            
            return f"检索成功（共找到{len(retrieved_docs)}个相关片段）：\n{context_str.strip()}"
        except Exception as e:
            return f"工具调用失败：{str(e)}"

    @tool
    def pdf_page_search_tool(page_num: str) -> str:
        """
        专门用于按页码查询PDF对应内容，支持单个页码（如「3」）和页码范围（如「5-8」），也支持带「第」「页」的表述（如「第4页」）。
        参数说明：
        - page_num：必填，要查询的页码（如「3」「5-8」「第6页」）
        """
        try:
            # 清理页码参数，仅保留数字和横杠
            clean_page = ''.join([c for c in page_num if c.isdigit() or c == '-'])
            if not clean_page:
                return "错误：请输入有效的页码（如「3」「5-8」）！"
            
            # 重新加载PDF文档
            all_docs = load_and_split_pdf()
            page_content = []
            
            # 处理页码范围
            if '-' in clean_page:
                start_page, end_page = clean_page.split('-')
                start_page = int(start_page) if start_page.isdigit() else 1
                end_page = int(end_page) if end_page.isdigit() else start_page
            else:
                start_page = end_page = int(clean_page) if clean_page.isdigit() else 1
            
            # 筛选对应页码的文档内容
            for doc in all_docs:
                doc_page = doc.metadata.get("page", "未知")
                try:
                    doc_page_int = int(doc_page)
                    if start_page <= doc_page_int <= end_page:
                        page_content.append(f"【页码{doc_page}】\n{doc.page_content}")
                except:
                    continue
            
            # 无结果时提示
            if not page_content:
                return f"未找到页码{page_num}对应的PDF内容，请确认页码有效。"
            
            # 格式化返回结果
            return f"页码{page_num}对应内容：\n{chr(10).join(page_content)}"
        except Exception as e:
            return f"工具调用失败：{str(e)}"

    # 工具列表（供Agent调用）
    tools = [pdf_meta_query_tool, pdf_vector_retrieve_tool, pdf_page_search_tool]

    # ========== Agent 配置（思考链+系统提示词，定义Agent行为逻辑） ==========
    # Agent 系统提示词（明确思考流程、工具使用规则、回答要求）
    system_prompt_str = """
    你是一个具备深度推理能力的PDF专业问答助手，仅可使用提供的3个工具解决问题，严格遵循以下流程：
    1.  第一步：先思考（必须执行，禁止跳过）
        请你先分析用户问题，拆解解决问题的步骤，并判断每一步是否需要调用工具：
        - 思考要点1：这个问题需要拆解成几个小步骤？每个步骤的目标是什么？
        - 思考要点2：每个步骤是否需要调用工具？如果需要，应该选择哪个工具？为什么选择这个工具？
        - 思考要点3：调用工具需要的参数是否完整？（如 pdf_vector_retrieve_tool 需要 query，pdf_page_search_tool 需要 page_num）
        - 思考要点4：如果工具调用失败，是否需要调整参数重试？（如检索不到内容时，是否需要扩展关键词）
    2.  第二步：工具调用（严格按思考结果执行）
        工具匹配规则（仅参考，需结合思考灵活调整）：
        - 当需要查询PDF的核心主题或5个核心关键词时，调用 pdf_meta_query_tool（无参数）
        - 当需要检索PDF中与具体问题（如业绩、数据、概念定义）相关的文本片段时，调用 pdf_vector_retrieve_tool（必填参数：query=用户完整问题；可选参数：top_k=2）
        - 当需要按页码/页码范围查询PDF具体内容时，调用 pdf_page_search_tool（必填参数：page_num=用户指定的页码/页码范围）
        - 当问题是闲聊、无关知识（如天气、计算）时，无需调用工具，直接返回明确回答
    3.  第三步：结果处理（可选：二次思考）
        - 若获取工具返回结果后，已能完整回答用户问题，直接汇总整理结果，保持简洁明了；
        - 若工具返回结果不完整，无法回答用户问题，需要再次思考：是否需要补充调用其他工具？或调整参数重新调用同一工具？
    4.  回答要求：
        - 严格基于工具返回结果生成回答，绝不编造任何信息；
        - 如果工具返回的检索片段中**没有相关内容**，直接回复「未检索到与营收增速相关的有效信息，请调整关键词重试」，禁止输出任何猜测性数据；
        - 保留关键溯源信息（如页码、检索片段编号），提升回答可信度；
        - 工具调用失败时，如实反馈失败原因（如“未检索到相关内容”“页码无效”），不强行回答；
    """

    # 创建Agent（LangChain 1.0+ 规范，无AgentExecutor）
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt_str,
    )

    # ========== 相关性判断函数（过滤无关问题，提升问答效率） ==========
    def is_question_relevant(question):
        q_normalized = normalize_question(question)
        
        # 1. 优先查自定义缓存，提升效率
        if q_normalized in relevance_cache:
            print(f"🔍 命中相关性缓存：{question}")
            return relevance_cache[q_normalized]
        
        # 2. 未命中缓存时，调用LLM判断相关性
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""请判断问题是否和以下主题相关：
            核心主题：{pdf_info['topic']}
            核心关键词：{','.join(pdf_info['keywords'])}
            相关：问题围绕PDF内容、主题、关键词展开
            非相关：天气、闲聊、无关知识、纯计算等
            输出格式：先写【相关】或【非相关】，空格后写理由"""),
            ("human", f"问题：{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        try:
            print(f"📡 调用LLM判断相关性：{question}")
            result = chain.invoke({}).strip()
            
            # 解析判断结果
            if result.startswith("【相关】"):
                is_rel = True
                reason = result.replace("【相关】", "").strip()
            else:
                is_rel = False
                reason = result.replace("【非相关】", "").strip() if "【非相关】" in result else "问题与PDF内容无关"
            
            # 存入缓存，后续复用
            relevance_cache[q_normalized] = (is_rel, reason)
            return is_rel, reason
        except Exception as e:
            print(f"⚠️ 判断相关性失败：{e}")
            return False, "无法判断相关性"

    # ========== 主问答函数（整合所有逻辑，对外提供问答接口） ==========
    def qa_function(question):
        # 空值校验
        question_clean = question.strip()
        if not question_clean:
            return "⚠️ 请输入有效的问题！", []
        
        # 标准化问题
        q_normalized = normalize_question(question_clean)
        
        # 1. 优先查最终问答结果缓存，提升效率
        if q_normalized in qa_cache:
            print(f"🔍 命中问答缓存：{question_clean}")
            return qa_cache[q_normalized]
        
        # 2. 判断问题相关性，过滤无关问题
        is_relevant, reason = is_question_relevant(question_clean)
        if not is_relevant:
            result = (f"❌ 抱歉，我仅能回答与PDF相关的问题哦～（原因：{reason}）", [])
            qa_cache[q_normalized] = result
            return result
        
        # 3. 获取分层记忆（短期+长期），提升对话连贯性
        combined_memory = memory_manager.get_combined_memory()

        # 4. 调用Agent工具链生成回答
        try:
            # 构造Agent输入
            inputs = {
                "messages": [
                    {"role": "user", "content": f"历史对话记忆：{combined_memory}\n用户问题：{question_clean}"}
                ]
            }
            answer = ""
            # 遍历Agent stream输出，提取思考过程和最终回答
            for chunk in agent.stream(inputs, stream_mode="updates"):
                if "model" in chunk and len(chunk["model"]["messages"]) > 0:
                    latest_msg = chunk["model"]["messages"][-1]
                    # 打印工具调用决策（方便调试和展示Agent思考过程）
                    if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                        for tool_call in latest_msg.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            tool_call_id = tool_call["id"]
                            print("=" * 30 + " Agent 思考过程 " + "=" * 30)
                            print(f"🤔 推理目标：解决用户问题「{question_clean}」")
                            print(f"✅ 决策结果：调用工具「{tool_name}」")
                            print(f"📋 工具参数：{tool_args}")
                            print(f"🆔 调用ID：{tool_call_id}")
                            print(f"📝 推理原因：该工具是获取「{question_clean}」相关信息的最优选择，可直接满足查询需求")
                            print("=" * 68)
                    
                    # 提取有效回答内容
                    if latest_msg.content.strip():
                        answer = latest_msg.content
                        # 打印Agent自然语言思考（可选，提升可解释性）
                        if "我需要先" in latest_msg.content or "第一步" in latest_msg.content:
                            print("=" * 30 + " Agent 语义思考 " + "=" * 30)
                            print(f"💡 自然语言思考：{latest_msg.content}")
                            print("=" * 68)
            
            # 兜底：Agent未返回有效回答时，手动触发工具调用
            if not answer.strip():
                print("⚠️ Agent未返回有效回答，手动触发工具调用")
                if any(word in question_clean for word in ["主题", "关键词", "讲了什么"]):
                    answer = pdf_meta_query_tool.invoke({})
                elif any(word in question_clean for word in ["页码", "第几页"]):
                    answer = "请明确输入要查询的页码（如「3」「5-8」）"
                else:
                    answer = pdf_vector_retrieve_tool.invoke({"query": question_clean, "top_k": 2})
            
            sources = ["工具调用结果"]
        except Exception as e:
            # Agent调用失败时，切换为普通RAG兜底
            print(f"⚠️ Agent执行失败，切换为普通RAG回答：{e}")
            retrieved_docs = retriever.invoke(question_clean)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            sources = [f"页码{doc.metadata.get('page', '未知')}" for doc in retrieved_docs] or ["无可用来源"]
            answer = answer_chain.invoke({
                "memory": combined_memory,
                "context": context,
                "question": question_clean
            }).strip()
        
        # 5. 更新分层记忆，保存当前对话
        memory_manager.add_message("user", question_clean)
        memory_manager.add_message("assistant", answer)
        
        # 6. 存入问答缓存，后续复用
        result = (answer, sources)
        qa_cache[q_normalized] = result
        
        return result

    # ========== 辅助函数（清空记忆/缓存，提升易用性） ==========
    def clear_history():
        """清空分层记忆（短期+长期）"""
        return memory_manager.clear_all()
    
    def clear_all_cache():
        """清空所有缓存（问答缓存+相关性缓存+LLM缓存）"""
        global qa_cache, relevance_cache
        qa_cache = {}
        relevance_cache = {}
        llm_cache.clear()
        return "🧹 所有缓存已清空"

    # 返回核心可调用函数
    return qa_function, clear_history, clear_all_cache

# ------------------------------ 主程序入口（程序启动入口） ------------------------------
if __name__ == "__main__":
    # 创建必要目录（缓存目录+向量库目录）
    os.makedirs("./cache", exist_ok=True)
    os.makedirs(CONFIG["chroma_dir"], exist_ok=True)
    
    try:
        # 初始化问答助手
        print("🚀 初始化PDF问答助手（分层记忆+自定义缓存+工具调用版）...")
        qa_func, clear_history_func, clear_all_cache_func = build_qa_chain()
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