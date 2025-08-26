# 法律条文 RAG 助手

一个面向 **中国劳动法问答** 的 **检索增强生成 (RAG)** 系统，集成：  
- **法律文本的数据收集与结构化**  
- **向量检索与重排序**  
- **大模型生成（附带引用）**  
- **LoRA 微调实现领域适配**  
- **评估体系**  
- **Streamlit 前端 + vLLM 部署**  

---

## 项目目标
- 构建一个 **智能法律助手**，能够：  
  - 支持法律条文的 **按月动态更新**  
  - 精准引用法律条款（如“《劳动法》第36条”）  
  - 处理 **复杂的多条款关联问题**  
- 确保答案具有 **可溯源性与可解释性**  
- 结合 **RAG 检索** 与 **LoRA 微调** 提升鲁棒性  

---

## 系统架构
项目分为 **六个层次**：

1. **数据收集与结构化**
   - 从政府官网爬取法规文本
   - 使用正则 (`第X条`) 解析条款
   - 存储为 JSON 格式：
     ```json
     {
       "中华人民共和国劳动法 第36条": "用人单位因生产经营需要...",
       "中华人民共和国劳动合同法 第10条": "建立劳动关系应当订立书面劳动合同..."
     }
     ```

2. **索引与向量存储**
   - 使用 `bge-small-zh` 进行向量化
   - 存储在 **ChromaDB**
   - 节点 ID 稳定化：`{filename}::{clause}`

3. **检索与重排序**
   - 初筛：语义检索 Top-10
   - 精排：交叉编码器 (`bge-reranker-large`) 取 Top-3

4. **生成层**
   - 大模型（Qwen1.5 / DeepSeek-R1 等）
   - Prompt 包含：用户问题 + 检索条款 + 指令模板（要求附引用）

5. **评估与优化**
   - 检索召回率提升：**63.5% → 87.3%**
   - 端到端评估指标：
     - 引用正确率
     - 内容相似度
     - 条件覆盖完整性
   - LoRA 微调提升主体识别：**61.2% → 93.7%**

6. **应用层**
   - **Streamlit 前端**：
     - 问答界面
     - 可展开条款引用
     - 检索深度可调
   - **vLLM 推理加速**：
     - 生成速度提升 7.1x
     - 显存占用降低 38%

---

## 安装

### 环境要求
- Python 3.9+
- 推荐 GPU + CUDA 11.8/12.x
- 已提供 `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 使用方法

### 1. 准备数据
```bash
python scripts/fetch_and_parse.py   --url "https://www.gov.cn/law/2005-05/25/content_905.htm"   --out data/processed/labor_law.json
```

或将已整理好的 JSON 放在 `data/processed/`。

### 2. 构建索引
```bash
python scripts/build_index.py   --data_dir data/processed   --vector_db_dir ./chroma_db   --persist_dir ./storage   --collection chinese_labor_laws   --embed_model "BAAI/bge-small-zh-v1.5"   --top_k 10
```

### 3. 启动 LLM 后端（可选，vLLM）
```bash
python -m vllm.entrypoints.openai.api_server   --model DeepSeek-R1-Distill-Qwen-1___5B   --port 8000
```

### 4. 运行 CLI 问答
```bash
python scripts/qa_cli.py
# 输入: 试用期被辞退如何获得补偿？
```

### 5. 启动 Streamlit 前端
```bash
streamlit run app/legal_assistant.py --server.port 8501
```

---

## 评估结果

### 检索评估
- Top-3 召回率：**87.3%**

### 端到端评估
- 引用条款是否正确
- 内容相似度（Jaccard / F1）
- 条件覆盖是否完整

### LoRA 微调
- 提升劳动仲裁等小众场景识别准确率
- 主体识别准确率从 **61.2% → 93.7%**

---

## 示例
**输入**: “试用期被辞退如何获得补偿？”  
**输出**:  
> 根据《劳动合同法》第二十一条规定，用人单位在试用期解除劳动合同的，应当说明理由。若违法解除，劳动者可依据第四十七条要求经济补偿。  

**引用来源**:  
- 《劳动合同法》第二十一条  
- 《劳动合同法》第四十七条  

---

## 展望
- 扩展至更多法律领域（不仅限于劳动法）  
- 增加多语言支持  
- 融入 **RAG-Fusion**、**Tree-of-Thought** 等推理方法  

---