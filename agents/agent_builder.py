from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import CONFIG
from cache.qa_cache import get_qa_cache, set_qa_cache, get_relevance_cache, set_relevance_cache
from memory.layered_memory import LayeredMemoryManager
from retrieval.pdf_loader import get_pdf_basic_info
from retrieval.hybrid_retriever import init_bm25_retriever
from .tools import get_agent_tools

def normalize_question(question):
    """
    标准化问题（统一格式，确保相同问题命中缓存）
    :param question: 原始问题
    :return: 标准化后的问题
    """
    return question.strip().lower()

def is_question_relevant(question, pdf_info, llm):
    """
    相关性判断函数（过滤无关问题，提升问答效率）
    :param question: 用户问题
    :param pdf_info: PDF信息
    :param llm: 语言模型
    :return: (是否相关, 理由)
    """
    q_normalized = normalize_question(question)
    
    # 1. 优先查自定义缓存，提升效率
    if q_normalized in get_relevance_cache():
        print(f"🔍 命中相关性缓存：{question}")
        return get_relevance_cache()[q_normalized]
    
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
        set_relevance_cache(q_normalized, (is_rel, reason))
        return is_rel, reason
    except Exception as e:
        print(f"⚠️ 判断相关性失败：{e}")
        return False, "无法判断相关性"

def build_qa_agent(llm, docs, embeddings, vector_db, retriever, pdf_hash, pdf_info, reranker):
    """
    构建问答Agent
    :param llm: 语言模型
    :param docs: 文档列表
    :param embeddings: 嵌入模型
    :param vector_db: 向量数据库
    :param retriever: 向量检索器
    :param pdf_hash: PDF哈希值
    :param pdf_info: PDF信息
    :param reranker: 重排序模型
    :return: 问答函数、清空历史函数、清空缓存函数
    """
    # 初始化BM25检索器
    bm25_retrieve = init_bm25_retriever(docs)

    # 初始化分层记忆（短期记忆+长期记忆，提升对话连贯性）
    memory_manager = LayeredMemoryManager(
        llm=llm,
        short_term_max_rounds=CONFIG["short_term_max_rounds"]
    )

    # 构建降级用回答链（Agent调用失败时，使用基础RAG兜底）
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

    # 获取Agent工具列表
    tools = get_agent_tools(pdf_info, bm25_retrieve, retriever, llm, reranker)

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

    # 主问答函数（整合所有逻辑，对外提供问答接口）
    def qa_function(question):
        # 空值校验
        question_clean = question.strip()
        if not question_clean:
            return "⚠️ 请输入有效的问题！", []
        
        # 标准化问题
        q_normalized = normalize_question(question_clean)
        
        # 1. 优先查最终问答结果缓存，提升效率
        if q_normalized in get_qa_cache():
            print(f"🔍 命中问答缓存：{question_clean}")
            return get_qa_cache()[q_normalized]
        
        # 2. 判断问题相关性，过滤无关问题
        is_relevant, reason = is_question_relevant(question_clean, pdf_info, llm)
        if not is_relevant:
            result = (f"❌ 抱歉，我仅能回答与PDF相关的问题哦～（原因：{reason}）", [])
            set_qa_cache(q_normalized, result)
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
                    answer = tools[0].invoke({})
                elif any(word in question_clean for word in ["页码", "第几页"]):
                    answer = "请明确输入要查询的页码（如「3」「5-8」）"
                else:
                    answer = tools[1].invoke({"query": question_clean, "top_k": 2})
            
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
        set_qa_cache(q_normalized, result)
        
        return result

    # 辅助函数（清空记忆/缓存，提升易用性）
    def clear_history():
        """清空分层记忆（短期+长期）"""
        return memory_manager.clear_all()
    
    def clear_all_cache():
        """清空所有缓存（问答缓存+相关性缓存+LLM缓存）"""
        from cache.qa_cache import clear_qa_cache, clear_relevance_cache, clear_all_cache as clear_all_cache_impl
        clear_qa_cache()
        clear_relevance_cache()
        clear_all_cache_impl()
        return "🧹 所有缓存已清空"

    # 返回核心可调用函数
    return qa_function, clear_history, clear_all_cache
