from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import chromadb

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

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
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "。", "！", "？", " ", ""]
)

chunks = text_splitter.split_documents(raw_documents)
# Load the tokenizer corresponding to your embedding model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

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
