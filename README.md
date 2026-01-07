# 📚 PDF智能问答助手

一个基于RAG、Agent和分层记忆的PDF智能问答系统，支持语义检索、多查询扩展、混合检索和工具调用。

## 🚀 功能演示
![功能演示](assets/demo.gif)

## 📊 项目介绍
本项目是一款面向复杂文档的**智能问答系统**，能够深度理解PDF文档内容，为用户提供精准、高效的自然语言问答服务。

用户无需手动翻阅文档，只需输入问题即可快速获取对应答案，同时支持多轮对话上下文理解，大幅提升专业文档的阅读和信息提取效率。

系统可广泛应用于金融、法律、医疗等需要处理大量专业PDF文档的场景，帮助企业或个人快速定位关键信息、降低信息检索成本、提升工作效率。

## ✨ 核心特性

### 🚀 智能检索增强
- **混合检索策略**: 结合BM25关键词检索与Chroma向量检索，兼顾精准性与泛化性
- **多查询扩展**: 自动生成3个同义查询，提升检索召回率
- **异步并行检索**: 多查询并行执行，抵消扩展带来的耗时增加
- **文档重排序**: 使用交叉编码器对检索结果精排，提升相关性

### 🧠 智能对话能力
- **分层记忆管理**: 短期记忆(最近对话) + 长期记忆(历史摘要)，保持对话连贯性
- **Agent工具调用**: 内置3个专业工具，覆盖PDF查询核心场景
- **智能相关性判断**: 自动过滤无关问题，专注PDF内容问答
- **多级缓存优化**: LLM缓存+业务缓存，提升重复查询效率

### 🛠️ 专业工具支持
1. **PDF元信息查询**: 获取文档主题和核心关键词
2. **语义向量检索**: 基于混合检索技术查找相关内容片段
3. **页码精准查询**: 支持单页或页码范围的内容定位

## 📁 项目结构

```
.
├── main.py                    # 主程序入口
├── cache_utils.py            # PDF缓存管理工具
├── layered_memory.py         # 分层记忆管理器
├── assets/
│   └── file.pdf              # 待处理的PDF文档
├── models/                   # 本地模型目录
│   ├── bge-small-zh-v1.5/    # 嵌入模型
│   └── bge-reranker-base/    # 重排序模型
├── chroma_db/               # Chroma向量数据库
└── .env.example             # 环境变量模板
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd <project-directory>

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型下载

```bash
# 创建模型目录
mkdir -p models

# 下载嵌入模型（约500MB）
git lfs install
git clone https://huggingface.co/BAAI/bge-small-zh-v1.5 models/bge-small-zh-v1.5

# 下载重排序模型（约500MB）
git clone https://huggingface.co/BAAI/bge-reranker-base models/bge-reranker-base
```

### 3. 配置文件

复制环境变量模板并配置API密钥：

```bash
cp .env.example .env
```

编辑`.env`文件：
```
# 通义千问API配置
QWEN_API_KEY=your_qwen_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 4. 准备PDF文档

将待处理的PDF文件放入`assets/`目录，或修改配置中的PDF路径：

```python
# 在CONFIG中修改
"pdf_path": "./assets/your_document.pdf"
```

### 5. 启动系统

```bash
python main.py
```

## 📖 使用方法

### 基本命令
```
请输入你的问题： [输入问题或命令]

命令说明：
  - quit：退出程序
  - clear：清空对话记忆
  - clear_cache：清空所有缓存
```

### 示例对话

```
🚀 初始化PDF问答助手（分层记忆+自定义缓存+工具调用版）...
✅ 助手就绪！
📖 命令说明：
  - quit：退出程序
  - clear：清空对话记忆
  - clear_cache：清空所有缓存
  - 支持工具调用：查主题/关键词、查具体内容、按页码查内容
------------------------------------------------------------

请输入你的问题：这份PDF主要讲了什么？

📝 回答：这份PDF的核心主题是...

请输入你的问题：第5页有什么内容？

📝 回答：页码5对应内容：【页码5】...

请输入你的问题：今年的营收增速是多少？

📝 回答：根据文档内容，今年营收增速为...
📎 来源：工具调用结果
```

## 🔧 配置说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_chunk_size` | 300 | 文档分块最大字符数 |
| `min_chunk_size` | 100 | 文档分块最小字符数 |
| `chunk_overlap` | 30 | 分块重叠字符数 |
| `retrieve_top_k` | 2 | 单次检索返回片段数 |
| `short_term_max_rounds` | 3 | 短期记忆最大轮数 |
| `llm_temperature` | 0.1 | LLM温度参数 |

### 模型配置

- **嵌入模型**: `BAAI/bge-small-zh-v1.5` (中文优化)
- **重排序模型**: `BAAI/bge-reranker-base`
- **LLM模型**: `qwen-plus` (可通过API配置切换)

## 🧠 技术架构

### 核心模块

```
1. 文档处理层
   ├── PDF加载与解析
   ├── 语义分块（段落+句子级）
   └── 向量化存储

2. 检索增强层
   ├── BM25关键词检索
   ├── Chroma向量检索
   ├── 多查询扩展
   └── 交叉编码器重排序

3. 对话管理层
   ├── 分层记忆系统
   ├── 相关性判断
   └── 多级缓存机制

4. Agent工具层
   ├── 元信息查询工具
   ├── 语义检索工具
   └── 页码查询工具
```

### 工作流程

```
用户提问 → 问题标准化 → 缓存检查 → 相关性判断 → 记忆管理 → Agent工具调用 → 结果生成 → 缓存存储
```

## 📊 性能优化

### 缓存策略
- **LLM原生缓存**: 内置LangChain缓存，减少重复API调用
- **业务级缓存**: 问答结果缓存 + 相关性判断缓存
- **PDF元信息缓存**: 避免重复提取文档主题和关键词

### 异步优化
- 多查询并行检索，提升召回率同时保持性能
- 同步包装器兼容现有架构，无需重构

### 内存管理
- 短期记忆限制轮数，防止上下文过长
- 长期记忆自动摘要，保留重要信息

## 🐛 常见问题

### Q: 如何处理大型PDF？
A: 系统采用语义分块策略，按段落和句子拆分，保证语义完整性。可通过调整`max_chunk_size`和`min_chunk_size`优化分块效果。

### Q: 支持哪些格式的文档？
A: 目前主要支持PDF格式，通过`llama-index`的`SimpleDirectoryReader`加载。

### Q: 如何切换其他LLM模型？
A: 修改`.env`文件中的API配置，或调整`llm_model`参数使用其他兼容OpenAI API的模型。

### Q: 检索效果不理想怎么办？
A: 可尝试以下方法：
1. 调整`retrieve_top_k`参数增加检索数量
2. 优化分块参数（`max_chunk_size`等）
3. 检查嵌入模型是否适合你的文档类型



## 📄 许可证

本项目采用Apache 2.0许可证。
