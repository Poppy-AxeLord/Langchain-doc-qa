from langchain_core.tools import tool

from retrieval.pdf_loader import load_and_split_pdf
from retrieval.hybrid_retriever import hybrid_retrieve

@tool
def pdf_meta_query_tool(pdf_info) -> str:
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
def pdf_vector_retrieve_tool(query: str, top_k: int = 2, bm25_retrieve=None, retriever=None, llm=None, reranker=None) -> str:
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
        retrieved_docs = hybrid_retrieve(
            query=query, 
            top_k=top_k, 
            llm=llm, 
            bm25_retrieve=bm25_retrieve, 
            retriever=retriever, 
            reranker=reranker
        )
        
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

def get_agent_tools(pdf_info, bm25_retrieve, retriever, llm, reranker):
    """
    获取Agent工具列表
    :param pdf_info: PDF信息
    :param bm25_retrieve: BM25检索函数
    :param retriever: 向量检索器
    :param llm: 语言模型
    :param reranker: 重排序模型
    :return: 工具列表
    """
    from langchain_core.tools import StructuredTool
    
    # 定义工具函数
    def pdf_meta_tool():
        """
        专门用于查询当前PDF的核心元信息（主题+5个关键词）。
        当用户提问类似「这份PDF讲了什么？」「PDF关键词有哪些？」「文档主题是什么？」时调用。
        输出返回PDF主题和关键词的原始信息，不添加任何额外话术、寒暄、引导提问等无关内容。
        """
        return pdf_meta_query_tool(pdf_info)
    
    def pdf_retrieve_tool(query: str, top_k: int = 2):
        """
        专门用于检索PDF中与用户问题相关的精准文本片段，用于回答具体概念、数据、方法等非主题类问题。
        触发条件：当用户提问不涉及PDF主题/关键词，也不涉及页码时调用。
        参数说明：
        - query：必填项，传入用户完整查询问题。
        - top_k：可选项，检索返回的文本片段数量，默认2。
        """
        return pdf_vector_retrieve_tool(
            query=query, 
            top_k=top_k, 
            bm25_retrieve=bm25_retrieve, 
            retriever=retriever, 
            llm=llm, 
            reranker=reranker
        )
    
    # 创建结构化工具
    meta_tool = StructuredTool.from_function(
        func=pdf_meta_tool,
        name="pdf_meta_query_tool",
        description="专门用于查询当前PDF的核心元信息（主题+5个关键词）"
    )
    
    retrieve_tool = StructuredTool.from_function(
        func=pdf_retrieve_tool,
        name="pdf_vector_retrieve_tool",
        description="专门用于检索PDF中与用户问题相关的精准文本片段"
    )
    
    return [meta_tool, retrieve_tool, pdf_page_search_tool]
