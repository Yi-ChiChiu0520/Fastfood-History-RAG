# Import required modules
import os  # For file and environment variable handling
from dotenv import load_dotenv  # To load environment variables from a .env file
import chromadb  # ChromaDB client library for vector database
from chromadb import Documents, EmbeddingFunction, Embeddings  # Typing and interfaces
from sentence_transformers import SentenceTransformer  # Pre-trained embedding model
from openai import OpenAI  # OpenAI API for GPT interaction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load environment variables from a .env file (e.g., your OpenAI API key)
load_dotenv()

# Set paths
DATA_PATH = r"data"  # Folder path to where your raw documents/PDFs are stored
CHROMA_PATH = r"chroma_db"  # Folder path for storing the Chroma vector DB

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# Define a custom embedding function using multilingual E5 model, which handles Traditional Chinese
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-large")  # Load model

    def __call__(self, input: Documents) -> Embeddings:
        # Prepend 'passage:' to each input string as expected by the E5 model
        return self.model.encode([f"passage: {text}" for text in input], show_progress_bar=False).tolist()


# Instantiate the embedding function
embedding_function = MyEmbeddingFunction()

# Either retrieve an existing Chroma collection or create a new one
collection = chroma_client.get_collection(
    name="my-collection",  # Name of the vector collection
    embedding_function=embedding_function  # Embedding function used for similarity search
)

# Create an OpenAI client using your API key from the environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"  # Use GPT-4o model for answering questions

# Start an infinite loop for user interaction (Q&A style)
while True:
    # Get the user's question from input
    user_query = input("請輸入你關於麥當勞歷史的問題（輸入 'Goodbye Adam' 結束）：\n")

    # Check if the user wants to end the session
    if user_query.strip().lower() == "goodbye adam":
        print("再見！很高興為你服務。")  # Farewell message
        break

    # Query the Chroma collection to find the top 3 most relevant chunks
    results = collection.query(
        query_texts=[user_query],  # The user's query
        n_results=5,  # Number of matching chunks to return
        include=["documents", "metadatas"]  # Include the actual texts and metadata
    )

    # Format and print the top 5 retrieved chunks for reference
    retrieved_chunks = ["\n".join(doc_list) for doc_list in results["documents"]]
    context = "\n\n".join(retrieved_chunks[:3])
    print("查詢到前五匹配的資料：", context)

    # Load reranker tokenizer and model
    reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    reranker_model.eval() # Set model to evaluation mode

    highest_score = float("-inf")
    best_chunk = ""
    best_id = ""

    print("開始計算匹配度分數...\n")

    # Rerank the 5 retrieved chunks using cross-encoder model
    for doc_id_list, doc_list in zip(results["ids"], results["documents"]):
        for doc_id, doc_text in zip(doc_id_list, doc_list):

            inputs = reranker_tokenizer(user_query, doc_text, return_tensors="pt", truncation=True)

            with torch.no_grad():
                output = reranker_model(**inputs)
                score = output.logits.item()

            print(f"ID: {doc_id}")
            print(f"Score: {score}")
            print(f"Text: {doc_text}")
            print("-" * 40)

            if score > highest_score:
                highest_score = score
                best_chunk = doc_text
                best_id = doc_id

    print(f"使用最匹配的資料：\n ID: {best_id} \n Score: {highest_score} \n Text: {best_chunk}")

    # Create a system prompt to guide GPT on how to behave
    system_prompt = f"""
    你的名字是 Adam。你是一位樂於助人的助手，負責回答有關台灣速食歷史和文化發展的問題。
    但你只能根據使用者提供的資訊來回答問題，不能使用你自己的內部知識，也不能憑空捏造內容。
    
    如果你不知道答案，就回答：「我不知道。」
    如果使用者說「Goodbye Adam」，你要以親切的告別訊息回覆對方。
    
    ------------------------
    
    以下是可用資料：
    {best_chunk}
    """

    # Use OpenAI's chat completion endpoint to generate a response from GPT-4o
    response = client.chat.completions.create(
        model=model,  # GPT-4o model
        messages=[
            {"role": "system", "content": system_prompt},  # System role defines behavior and context
            {"role": "user", "content": user_query}  # User input to which GPT should respond
        ]
    )

    # Print the final answer from GPT
    print("\n\n--------------------\n\n")
    print("Adam 的回答：", response.choices[0].message.content)
