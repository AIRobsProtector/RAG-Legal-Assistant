# Legal Document RAG Assistant

A **Retrieval-Augmented Generation (RAG)** system for **Chinese labor law Q&A**, combining:
- **Data collection & structuring** of legal texts
- **Vector search & re-ranking**
- **LLM generation with citation**
- **LoRA fine-tuning for domain adaptation**
- **Evaluation framework**
- **Streamlit UI + vLLM deployment**

---

## Project Goals
- Build a **legal assistant** capable of:
  - Monthly updates of newly released laws
  - Precise citation (e.g., “《劳动法》第36条”)
  - Handling **complex multi-clause queries**
- Ensure **traceability & interpretability** of answers
- Combine **RAG retrieval** and **LoRA fine-tuning** for robustness

---

## System Architecture
The project consists of **six layers**:

1. **Data Collection & Structuring**
   - Crawl from official sites
   - Parse legal texts with regex (`第X条`)
   - Store structured JSON format:
     ```json
     {
       "中华人民共和国劳动法 第36条": "用人单位因生产经营需要...",
       "中华人民共和国劳动合同法 第10条": "建立劳动关系应当订立书面劳动合同..."
     }
     ```

2. **Indexing & Vector Store**
   - Embedding with `bge-small-zh`
   - Storage with **ChromaDB**
   - Nodes have stable IDs (`{filename}::{clause}`)

3. **Retrieval & Re-ranking**
   - Initial semantic retrieval (Top-10)
   - Cross-encoder reranker (`bge-reranker-large`) → Top-3

4. **Generation Layer**
   - LLM (Qwen1.5 / DeepSeek-R1 etc.)
   - Prompts include:
     - User question
     - Retrieved clauses
     - Instruction template (“Answer with citations”)

5. **Evaluation & Optimization**
   - Retrieval recall tests (improved from **63.5% → 87.3%**)
   - End-to-end metrics:
     - Clause hit rate
     - Content similarity
     - Completeness of conditions
   - LoRA fine-tuning for subject recognition (accuracy **61.2% → 93.7%**)

6. **Application Layer**
   - **Streamlit frontend**:
     - User Q&A interface
     - Expandable cited clauses
     - Debug panel for retrieval depth
   - **vLLM backend**:
     - 7.1x faster inference, -38% memory

---

## Installation

### Requirements
- Python 3.9+
- GPU with CUDA 11.8/12.x (recommended)
- `requirements.txt` provided

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Data
```bash
python src/get_data/get_data.py   --url "https://www.gov.cn/law/2005-05/25/content_905.htm"   --out data/processed/labor_law.json
```

Or place your own structured JSON under `data/processed/`.

### 2. Build Index
```bash
python src/get_data/get_data.py   --data_dir data   --vector_db_dir ./chroma_db   --persist_dir ./storage  
```

### 3. Start LLM Backend (optional, via vLLM)
```bash
python -m vllm.entrypoints.openai.api_server   --model Qwen1___5-1___8B-Chat   --port 8000
```

### 4. Run QA
```bash
python src/deploy/run.py
# 输入问题: 试用期被辞退如何获得补偿？
```

### 5. Launch Streamlit UI
```bash
streamlit opens a new browser window, enter localhost:8501 to open the front-end interface
```

---

## Evaluation

### Retrieval Recall Benchmark
- Top-3 recall improved to **87.3%**

### End-to-End Evaluation
- Clause citation correctness
- Content similarity (Jaccard / F1)
- Completeness of mandatory conditions

### LoRA Fine-tuning
- Improves domain-specific subject recognition
- Example: distinguish employer vs employee

---

## Example Query
**Input**: “试用期被辞退如何获得补偿？”  
**Output**:
> According to **Article 21 of the Labor Contract Law**, the employer must provide justification when terminating during probation. If unlawful, compensation shall be provided under **Article 47**.  

**Cited Sources**:
- 《劳动合同法》第二十一条  
- 《劳动合同法》第四十七条

---

## Future Work
- Expand beyond labor law to more legal domains
- Add multilingual support
- Integrate **RAG-Fusion** or **Tree-of-Thought** reasoning

---