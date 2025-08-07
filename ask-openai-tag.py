# Import required modules
import os
import sqlite3
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
# Load environment variables
load_dotenv()

# Define constants
ALLOWED_TAGS = [
    "速食文化", "外食趨勢", "家庭結構轉變", "社會轉型", "家庭價值", "外食文化", "餐飲業發展",
    "麥當勞影響", "飲食文化轉變", "臺灣餐飲業發展", "麥當勞", "飲食全球化", "商標之爭",
    "臺灣速食文化", "在地化", "臺灣外食", "速食市場", "美式餐飲", "連鎖店展店",
    "飲食趨勢", "美式速食", "連鎖經營", "中式速食", "餐飲轉型", "速食風潮",
    "調理包技術", "早餐店崛起", "慢食運動", "臺灣飲食變遷", "生態環境", "飲食平衡"
]

# Set up OpenAI and reranker model
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
reranker_model.eval()

# Start interaction loop
while True:
    user_query = input("請輸入你關於麥當勞歷史的問題（輸入 'Goodbye Adam' 結束）：\n")

    if user_query.strip().lower() == "goodbye adam":
        print("再見！很高興為你服務。")
        break

    # Step 1: Rewrite query
    rewrite_prompt = [
        {
            "role": "system",
            "content": "你是一個負責幫忙重寫使用者查詢的助手，目的是幫助搜尋系統更容易理解問題。請將輸入的問題改寫為清晰、無錯字、適合檢索的格式，但不要改變原本語意。"
        },
        {"role": "user", "content": user_query}
    ]
    rewritten_response = client.chat.completions.create(model=model, messages=rewrite_prompt)
    cleaned_query = rewritten_response.choices[0].message.content.strip()
    print(f"\n📘 改寫後的查詢：{cleaned_query}\n")

    # Step 2: Generate tags
    tag_prompt = f"""
請根據下列問題，從預先定義的主題標籤中選出最多五個最相關的標籤。每個標籤必須來自下方提供的列表，且使用「、」分隔，不得創造新的標籤。若找不到適合的標籤，請回答「無」。

問題：
{cleaned_query}

可選標籤列表：
{"、".join(ALLOWED_TAGS)}

輸出格式：
標籤：標籤1、標籤2、標籤3
"""
    tag_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": tag_prompt}],
        temperature=0.5,
    )
    tag_output = tag_response.choices[0].message.content.strip()
    if "標籤：" in tag_output:
        predicted_tags = tag_output.split("標籤：")[-1].strip().split("、")
    else:
        predicted_tags = []

    print(f"\n📌 LLM 分析後的標籤：{predicted_tags}")

    # Step 3: Lookup chunks from SQLite DB
    conn = sqlite3.connect("chunks_with_tags.db")
    cursor = conn.cursor()
    matched_chunks_with_tags: dict[str, str]
    matched_chunks = []
    for tag in predicted_tags:
        cursor.execute("SELECT id, chunk, tags FROM chunks WHERE tags LIKE ?", [f"%{tag}%"])
        matched_chunks += cursor.fetchall()

    # Deduplicate and keep only top 5
    seen = set()
    chunk_matches = []
    for cid, chunk, tag_str in matched_chunks:
        if cid in seen:
            continue
        seen.add(cid)
        try:
            chunk_tags = json.loads(tag_str)
        except json.JSONDecodeError:
            chunk_tags = []

        matched_tags = list(set(predicted_tags) & set(chunk_tags))
        if matched_tags:
            chunk_matches.append({
                "id": cid,
                "chunk": chunk,
                "tags": chunk_tags,
                "matched_tags": matched_tags
            })


    conn.close()

    if not chunk_matches:
        print("❌ 找不到任何相關段落。")
        continue

    # ✅ Print matched chunks and tags
    print("\n🔍 符合標籤的段落及匹配標籤：\n")
    for i, match in enumerate(chunk_matches, start=1):
        print(f"Chunk ID: {match['id']}")
        print(f"Matched Tags: {match['matched_tags']}")
        print(f"All Tags: {match['tags']}")
        print(f"Text: {match['chunk'][:300]}...\n{'-' * 40}")

    # Step 4: Rerank
    print("\n🎯 開始進行交叉編碼重排序...\n")
    highest_score = float("-inf")
    best_chunk = ""
    best_id = ""

    for match in chunk_matches:  # limit to 5 if needed
        cid = match['id']
        doc_text = match['chunk']

        inputs = reranker_tokenizer(user_query, doc_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = reranker_model(**inputs)
            score = output.logits.item()

        print(f"ID: {cid} | Score: {score}")
        print(f"Text: {doc_text[:150]}...\n{'-' * 40}")

        if score > highest_score:
            highest_score = score
            best_chunk = doc_text
            best_id = cid

    print(f"\n✅ 使用最匹配的資料：ID={best_id}, Score={highest_score}")

    # Step 5: Final system prompt and LLM response
    system_prompt = f"""
你的名字是 Adam。你是一位樂於助人的助手，負責回答有關台灣速食歷史和文化發展的問題。
但你只能根據使用者提供的資訊來回答問題，不能使用你自己的內部知識，也不能憑空捏造內容。

如果你不知道答案，就回答：「我不知道。」
如果使用者說「Goodbye Adam」，你要以親切的告別訊息回覆對方。

------------------------

以下是可用資料：
{best_chunk}
"""
    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    print("\n\n--------------------\n\n")
    print("Adam 的回答：", final_response.choices[0].message.content)
