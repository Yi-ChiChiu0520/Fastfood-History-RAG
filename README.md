# 🍔 台灣速食文化 RAG 智慧助手

本專案是一個結合 **Retrieval-Augmented Generation (RAG)** 架構的智慧助手，讓使用者能深入探索 **台灣速食與麥當勞發展的歷史與文化**。本系統整合 **OpenAI GPT-4o** 與 **Meta LLaMA3.1-8B** 模型，結合嵌入式查詢、向量資料庫與交叉編碼器重排序（reranker）來提供準確答案。

---

## ✨ 功能亮點

- 🔍 **語意搜尋**：使用多語言句子嵌入模型，從本地 PDF 知識庫中擷取相關段落
- 🧠 **雙大型語言模型支援**：使用者可選擇 GPT-4o 或 LLaMA3.1-8B 來產出答案
- ⚖️ **雙階段檢索流程**：
  - 使用 Bi-Encoder 進行初步向量相似度篩選
  - 再透過 Cross-Encoder Reranker (`BAAI/bge-reranker-base`) 精準排序
- 🈶 **繁體中文支援**：基於 `intfloat/multilingual-e5-large` 模型訓練之嵌入向量
- 🗃️ **本地知識來源**：PDF 文件包含台灣速食與麥當勞的歷史資料

---

## ⚠️ 注意事項

- 必須具備 OpenAI API 金鑰（儲存於 `.env` 檔中）
- 執行前需安裝 Python 套件（見 `requirements.txt`）
- 若使用 LLaMA 模型，需配置正確的本地端推理環境或 API 端點

---

## 📁 檔案與功能說明

| 檔案名稱 | 說明                                                                     |
|----------|------------------------------------------------------------------------|
| `ask-llama.py` | 使用本地 LLaMA 模型來回答使用者問題與 `intfloat/multilingual-e5-large` 的嵌入模型做向量查詢流程   |
| `ask-openai.py` | 使用 OpenAI GPT-4o 回答問題與 `intfloat/multilingual-e5-large` 的嵌入模型做向量查詢流程   |
| `ask-openai-chroma.py` | 結合 OpenAI 與 ChromaDB 預設的 `all-MiniLM-L6-v2` 嵌入模型做向量查詢流程                |
| `ask-llama-chroma.py` | 使用 LLaMA 模型與 ChromaDB 預設的 `all-MiniLM-L6-v2` 嵌入模型做向量查詢流程               |
| `fill_db.py` | 將 PDF 文件切分為段落並存入 Chroma 向量資料庫，使用`intfloat/multilingual-e5-large` 的嵌入模型 |
| `fill_db_chromadb-embedding.py` | 將 PDF 文件切分為段落並存入 Chroma 向量資料庫，使用ChromaDB 預設的 `all-MiniLM-L6-v2` 嵌入模型                                 |
| `chroma_db/` | 儲存 Chroma 的向量資料（parquet 檔）                                             |
| `data/` | 放置原始 PDF 文件資料                                                          |
| `requirements.txt` | 所有需要安裝的 Python 套件列表                                                    |
| `.env`（需自行建立） | 儲存 OpenAI API 金鑰，如：`OPENAI_API_KEY=xxx`                                |

---

## 💻 安裝與執行方式

```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows 用戶：.venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt

# 設定 OpenAI 金鑰
echo "OPENAI_API_KEY=你的金鑰" > .env

# 建立向量資料庫
python fill_db_chromadb-embedding.py

# 提問範例
python ask-openai-chroma.py
