import asyncio
import re
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import jieba
from rank_bm25 import BM25Okapi

from config.settings import CONFIG

def generate_multi_queries(original_query, llm):
    """
    多查询生成核心函数（生成同义查询，提升检索召回率）
    :param original_query: 原始查询
    :param llm: 语言模型
    :return: 生成的查询列表
    """
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

def init_bm25_retriever(docs):
    """
    初始化BM25检索器
    :param docs: 文档列表
    :return: BM25检索函数
    """
    # 1. 提取所有文档片段的文本和元数据
    doc_texts = [doc.page_content for doc in docs]
    doc_metadata = [doc.metadata for doc in docs]

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

    return bm25_retrieve

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

def rerank_docs(query, retrieved_docs, reranker):
    """
    文档重排序函数（对检索结果精排，提升相关性排序准确性）
    :param query: 用户查询
    :param retrieved_docs: 混合检索得到的 Document 列表
    :param reranker: 重排序模型
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

def hybrid_retrieve(query: str, top_k: int = 2, llm=None, bm25_retrieve=None, retriever=None, reranker=None) -> list[Document]:
    """
    混合检索函数（BM25关键词检索 + Chroma向量检索）
    核心功能：融合关键词匹配的精准性和语义匹配的泛化性，提升检索召回率和精准度
    :param query: 用户查询语句
    :param top_k: 单检索策略返回的顶部片段数量
    :param llm: 语言模型
    :param bm25_retrieve: BM25检索函数
    :param retriever: 向量检索器
    :param reranker: 重排序模型
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
    if reranker:
        final_retrieved_docs = rerank_docs(query, final_retrieved_docs, reranker)

    return final_retrieved_docs
