# qa_app.py
import os
from dotenv import load_dotenv
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from retriever import DocRetriever

# ------------- Setup -------------
load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Chroma client & collection
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode([f"passage: {text}" for text in input], show_progress_bar=False).tolist()

embedding_function = MyEmbeddingFunction()

collection = chroma_client.get_collection(
    name="my-collection",
    embedding_function=embedding_function,
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o"

# Create our document-level retriever
retriever = DocRetriever(collection=collection)

# ------------- Loop -------------
print("🔎 文件層級檢索已啟動（輸入 'Goodbye Adam' 離開）")
while True:
    user_query = input("\n請輸入你的問題：\n").strip()
    if user_query.lower() == "goodbye adam":
        print("再見！很高興為你服務。")
        break

    # (Optional) rewrite query with GPT for cleaner retrieval
    rewriting_prompt = [
        {"role": "system", "content": "你是查詢改寫助手，請將輸入改寫為清晰、適合檢索但不改變語意的句子。"},
        {"role": "user", "content": user_query},
    ]
    rewritten = client.chat.completions.create(model=MODEL, messages=rewriting_prompt)
    cleaned_query = rewritten.choices[0].message.content.strip()
    print(f"\n📘 改寫後查詢：{cleaned_query}")

    # Document-level retrieve → rerank → aggregate
    top = retriever.retrieve_docs(
        query=cleaned_query,
        k_chunks=50,
        top_docs=5,
        evidences_per_doc=3,
        top_m_for_score=3,
    )

    # Show what we found
    context = retriever.build_context(top, max_chars=2500)
    print("\n📚 取得的文件與證據：\n" + context)

    # Compose an answering prompt using multi-doc evidence (not just a single chunk)
    system_prompt = f"""
你是 Adam，只能根據提供的資料回答，不能憑空捏造。
若無答案請回覆「我不知道」。

以下為可用資料（文件層級彙整的片段）：
----------------
{context}
----------------
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )

    print("\n--------------------\n")
    print("Adam 的回答：", response.choices[0].message.content)
