from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import sqlite3
import json

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"  # Use GPT-4o model for answering questions

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-large")

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode([f"passage: {text}" for text in input], show_progress_bar=False).tolist()


embedding_function = MyEmbeddingFunction()


collection = chroma_client.get_or_create_collection(
    name="my-collection",
    embedding_function=embedding_function
)

loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

for doc in raw_documents:
    doc.page_content = doc.page_content.replace("\n", "")

print(f"Loaded {len(raw_documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=250,
    separators=["\n\n", "\n", ".", "。", "！", "？", " ", ""]
)

chunks = text_splitter.split_documents(raw_documents)
# Load the tokenizer corresponding to your embedding model
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

documents = []
metadata = []
ids = []
token_counts = []

for i, chunk in enumerate(chunks):
    text = chunk.page_content
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_counts.append(len(tokens))

    documents.append(text)
    ids.append("ID" + str(i))
    metadata.append(chunk.metadata)


collection.upsert(
    documents=documents,
    metadatas=metadata,
    ids=ids,
)

data = collection.query(
    query_texts=["my query"],
    include=["documents", "metadatas", "embeddings"],
)

if data["embeddings"]:
    first_embedding = data["embeddings"][0][0]  # first list of embeddings, first embedding vector
    print("Embedding dimension:", len(first_embedding))
else:
    print("No embeddings returned.")

# print("Sample token counts per chunk:", token_counts)
# print("Average token count per chunk:", sum(token_counts) / len(token_counts))
# print("Max token count:", max(token_counts))
#
# print("\nAll Chunks:\n")
# for idx, doc in enumerate(documents):
#     print(f"Chunk {idx+1}:\n{doc}\n{'-'*40}")
#
# print("Inserted chunks into ChromaDB:", len(documents))


def generate_tags(text: str) -> list[str]:
    prompt = f"""
請閱讀以下段落，為其產生最多五個主題標籤，使用繁體中文，每個標籤不超過五個字，標籤之間請以「、」分隔。若不需要五個，請只提供必要數量。

段落：
{text}

請輸出格式如下：
標籤：標籤1、標籤2、標籤3
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        raw_output = response.choices[0].message.content.strip()
        if "標籤：" in raw_output:
            tags_line = raw_output.split("標籤：")[-1].strip()
            return tags_line.split("、")
        return []
    except Exception as e:
        print("Error generating tags:", e)
        return []

# Add tags for each chunk
all_tags = []

print("\nAll Chunks with tags:\n")
for idx, doc in enumerate(documents):
    tags = generate_tags(doc)
    all_tags.append(tags)
    print(f"Chunk {idx+1}, Tags: {', '.join(tags)}:\n{doc}\n{'-'*40}")

# Create (or connect to) database
conn = sqlite3.connect("chunks_with_tags.db")
cursor = conn.cursor()

# Create the table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    chunk TEXT NOT NULL,
    tags TEXT
)
""")

# Clear existing records (optional — if re-running for clean insert)
cursor.execute("DELETE FROM chunks")

# Insert each chunk and its tags
for idx, (chunk, tags) in enumerate(zip(documents, all_tags)):
    tags_json = json.dumps(tags, ensure_ascii=False)  # keep Chinese chars
    cursor.execute(
        "INSERT INTO chunks (id, chunk, tags) VALUES (?, ?, ?)",
        (idx + 1, chunk, tags_json)
    )

# Commit and close connection
conn.commit()
conn.close()

print(f"\n✅ Stored {len(all_tags)} chunks into 'chunks_with_tags.db'")