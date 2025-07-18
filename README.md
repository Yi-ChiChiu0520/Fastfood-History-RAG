# 🍔 Taiwan's Fastfood History RAG Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** assistant that helps users explore the **history and cultural evolution of fast food in Taiwan**, powered by both **OpenAI GPT-4o** and **Meta LLaMA3-8B**, combining the strength of multiple embedding models, ChromaDB, and cross-encoder reranking.

---

## ✨ Key Capabilities

- 🔍 **Semantic Search**: Retrieve relevant chunks from a PDF-based knowledge base using **multilingual sentence embeddings**.
- 🧠 **Dual LLM Support**: Choose between **GPT-4o** and **LLaMA3.1:8B** to generate grounded answers.
- ⚖️ **Two-Step Retrieval Pipeline**:
  - **Bi-encoder** for efficient similarity-based initial retrieval
  - **Cross-encoder reranker** (`BAAI/bge-reranker-base`) for improved relevance scoring
- 🗃️ **Knowledge Source**: Local PDF documents about fast food and McDonald's development in Taiwan
- 🈶 **Traditional Chinese Support** using `intfloat/multilingual-e5-large` for embedding and query understanding

---

## 🧩 Stack

- 📚 **LangChain** – document parsing and chunking
- 📦 **ChromaDB** – lightweight vector database for fast similarity search
- 🧠 **SentenceTransformers** – `intfloat/multilingual-e5-large` for chunk embeddings
- 🔁 **Hugging Face Transformers** – `BAAI/bge-reranker-base` for cross-encoder reranking
- 🤖 **LLMs**:
  - `gpt-4o` via OpenAI API
  - `LLaMA3-8B-Instruct` (local inference or via API)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/McDonaldsHistory-Agent.git
cd McDonaldsHistory-Agent
```

## 🛠 Setup Instructions
Create a Virtual Environment and Install Dependencies：
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## 🔐 Create a .env File
In the root directory, add your OpenAI API key:
``` bash
OPENAI_API_KEY=your-openai-api-key
```
## 💡 How It Works
1. Load and split your PDF documents (related to McDonald’s) into smaller chunks.

2. Store chunks in ChromaDB using multilingual embeddings.

3. On user query:
   - Retrieve top 5 candidates using vector similarity. 
   - Rerank them using a cross-encoder reranker. 
   - Use the best-matching chunk as context for GPT-4o to generate the final answer.
## 🧪 Example Query
``` bash
"請問麥當勞什麼時候在台灣開了第一間店？
 ----------------
 Adam 的回答（根據文件內容）：
「根據資料，麥當勞在1984年首次進入台灣市場，第一家店位於台北市。」"
```