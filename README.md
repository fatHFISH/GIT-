# 校园智能助手 (叶同学 RAG)

这是一个基于 **Qwen-Plus** 和 **FAISS** 实现的检索增强生成（RAG）系统。

### 🚀 项目亮点
- **QA解耦检索架构**：针对 RAG 中常见的检索偏离问题，本项目将检索项（问题）与内容项（完整QA）分离。通过对纯净问题进行向量化，显著提升了检索的 Top-1 准确率，避免了长文回答产生的语义噪声。
- **高性能向量搜索**：利用 FAISS `IndexFlatL2` 实现高效的本地语义匹配。
- **流式思考输出**：集成了通义千问的 `reasoning_content`，展示模型的思考过程。

### 🛠️ 技术栈
- **LLM**: Qwen-Plus (Alibaba Cloud)
- **Embedding**: `shibing624/text2vec-base-chinese`
- **Vector DB**: FAISS
- **Processing**: Regular Expression (Regex), Pickle

### 📦 快速启动
1. 安装依赖：`pip install -r requirements.txt`
2. 配置环境变量：`export DASHSCOPE_API_KEY='你的KEY'`
3. 构建索引：`python src/build_index.py`
4. 启动助手：`python src/chatbot.py`