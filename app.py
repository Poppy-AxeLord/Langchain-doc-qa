import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from llama_index.core import SimpleDirectoryReader
import os
from dotenv import load_dotenv

# 运行: streamlit run app.py

# ======================
# Streamlit页面配置（可视化基础）
# ======================
st.set_page_config(
    page_title="PDF智能问答助手",  # 页面标题
    page_icon="📄",                # 页面图标
    layout="wide"                  # 宽屏布局
)

# 页面标题+说明
st.title("📄 RAG PDF智能问答助手（通义千问版）")
st.divider()

# ======================
# 加载环境变量+基础配置
# ======================
load_dotenv()

# 配置项（和你的原代码一致）
EMBEDDING_PATH = "./models/bge-small-zh-v1.5"
CHROMA_DIR = "./chroma_db"
LLM_MODEL = "qwen-plus"
TEMPERATURE = 0.1

# ======================
# 侧边栏：配置+PDF上传
# ======================
with st.sidebar:
    st.header("⚙️ 配置")
    # 手动输入API密钥（可选，覆盖.env）
    qwen_api_key = st.text_input("通义千问API密钥", type="password", value=os.getenv("QWEN_API_KEY", ""))
    qwen_base_url = st.text_input("API地址", value=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    
    st.divider()
    
    # PDF上传（核心：支持上传任意PDF，替代固定demo.pdf）
    st.header("📤 上传PDF")
    uploaded_file = st.file_uploader("选择PDF文件", type="pdf")

# ======================
# 初始化核心组件（缓存避免重复加载）
# ======================
@st.cache_resource  # 缓存资源，避免每次刷新重新初始化
def init_components(uploaded_pdf):
    """初始化Embedding+LLM+向量库"""
    # 1. 保存上传的PDF到本地
    pdf_path = "./temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    
    # 2. 初始化Embedding（和你的原代码一致）
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 3. 初始化LLM（和你的原代码一致）
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=qwen_api_key or os.getenv("QWEN_API_KEY"),
        base_url=qwen_base_url or os.getenv("QWEN_BASE_URL")
    )
    
    # 4. 加载PDF+构建向量库（和你的原代码一致）
    llama_docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    docs = [
        Document(page_content=doc.text.strip(), metadata={"page": doc.metadata.get("page_label", "未知")})
        for doc in llama_docs
    ]
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    # 5. 构建RAG链（和你的原代码一致）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是中文智能文档助手，仅基于以下文档内容回答问题，答案简洁明了：\n{context}"),
        ("human", "{question}")
    ])
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# ======================
# 核心逻辑：只有上传PDF后才初始化
# ======================
if uploaded_file:
    # 初始化组件
    with st.spinner("🔧 正在加载PDF并初始化问答引擎..."):
        rag_chain, retriever = init_components(uploaded_file)
    st.success("✅ PDF加载完成，问答引擎已就绪！")
    
    st.divider()
    
    # ======================
    # 问答区域
    # ======================
    st.header("💬 智能问答")
    question = st.text_input("请输入你的问题（例如：文档中提到了什么内容？）")
    
    if st.button("提交问题", type="primary") and question:
        # 1. 检索相关文档（和你的原代码一致）
        with st.expander("📌 检索到的相关文档（LLM回答依据）", expanded=True):
            retrieved_docs = retriever.invoke(question)
            for i, doc in enumerate(retrieved_docs):
                st.write(f"### 相关文档 {i+1}")
                st.write(f"**页码**：{doc.metadata['page']}")
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                st.write(f"**内容**：{content}")
        
        # 2. 生成回答（和你的原代码一致）
        with st.spinner("🤔 正在生成回答..."):
            answer = rag_chain.invoke(question)
        
        # 3. 展示回答
        st.subheader("📝 回答")
        st.write(answer.strip())
        
        # 4. 展示来源
        st.subheader("📎 回答来源")
        sources = [f"页码{doc.metadata['page']}" for doc in retrieved_docs]
        st.write(f"来自：{', '.join(sources)}")

else:
    # 未上传PDF时的提示
    st.info("请在左侧侧边栏上传PDF文件，并配置通义千问API密钥，即可开始问答！")