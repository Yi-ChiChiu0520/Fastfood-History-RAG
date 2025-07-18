import os
from dotenv import load_dotenv
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI

# Load .env variables
load_dotenv()

# Paths
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# Get or create collection with embedding function
collection = chroma_client.get_or_create_collection(
    name="my-collection-chroma",
)

# Initialize OpenAI GPT-4o client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"

# Interactive Q&A loop
while True:
    user_query = input("請輸入你關於麥當勞歷史的問題（輸入 'Goodbye Adam' 結束）：\n")

    if user_query.strip().lower() == "goodbye adam":
        print("再見！很高興為你服務。")
        break

    # Query ChromaDB
    results = collection.query(
        query_texts=[user_query],
        n_results=1
    )

    print("查詢到的資料：", results['documents'])
    print("元資料：", results['metadatas'])

    # Prepare system prompt
    system_prompt = f"""
你的名字是 Adam。你是一位樂於助人的助手，負責回答有關台灣速食歷史和文化發展的問題。但你只能根據使用者提供的資訊來回答問題，不能使用你自己的內部知識，也不能憑空捏造內容。

如果你不知道答案，就回答：「我不知道。」
如果使用者說「Goodbye Adam」，你要以親切的告別訊息回覆對方。

------------------------

以下是可用資料：
{results['documents']}
    """

    # OpenAI chat completion
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    print("\n\n--------------------\n\n")
    print("Adam 的回答：", response.choices[0].message.content)
