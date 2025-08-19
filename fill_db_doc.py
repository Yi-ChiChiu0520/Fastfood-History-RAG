from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import os
import chromadb
import ssl
import nltk

# Fix SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"NLTK download warning (may still work): {e}")

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
    name="my-collection-original",
    embedding_function=embedding_function
)

# Load documents from multiple formats
raw_documents = []

# Load PDF files
pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
pdf_documents = pdf_loader.load()
raw_documents.extend(pdf_documents)
print(f"Loaded {len(pdf_documents)} PDF documents")

# Load DOCX files (modern Word format)
doc_documents = []
try:
    docx_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        silent_errors=True  # Skip files that can't be processed
    )
    docx_documents = docx_loader.load()
    doc_documents.extend(docx_documents)
    print(f"Loaded {len(docx_documents)} DOCX documents")
except Exception as e:
    print(f"Error loading DOCX files: {e}")


# Custom function to handle .doc files with multiple methods
def load_doc_files_manually():
    """Attempt to load .doc files using alternative methods"""
    doc_files = []
    try:
        import glob
        doc_file_paths = glob.glob(os.path.join(DATA_PATH, "**/*.doc"), recursive=True)

        if not doc_file_paths:
            print("No .doc files found")
            return doc_files

        print(f"Found {len(doc_file_paths)} .doc files to process")

        for file_path in doc_file_paths:
            print(f"Attempting to load .doc file: {os.path.basename(file_path)}")
            success = False

            # Method 1: Try UnstructuredWordDocumentLoader (requires LibreOffice)
            try:
                loader = UnstructuredWordDocumentLoader(
                    file_path,
                    mode="single"  # Use single mode to avoid complex parsing
                )
                documents = loader.load()
                if documents and documents[0].page_content.strip():
                    doc_files.extend(documents)
                    print(
                        f"✓ Successfully loaded .doc file with UnstructuredWordDocumentLoader: {os.path.basename(file_path)}")
                    success = True
                    continue
            except Exception as e:
                print(f"  - UnstructuredWordDocumentLoader failed: {str(e)[:100]}...")

    except Exception as e:
        print(f"Error in doc file processing: {e}")

    return doc_files


# Try to load .doc files manually
manual_doc_files = load_doc_files_manually()
doc_documents.extend(manual_doc_files)

raw_documents.extend(doc_documents)

print(f"Total loaded documents: {len(raw_documents)}")

# Clean up the documents
for doc in raw_documents:
    doc.page_content = doc.page_content.replace("\n", "")

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
    print(f"Chunk ID: {i} \n Text: {chunk}")

collection.upsert(
    documents=documents,
    metadatas=metadata,
    ids=ids,
)

data = collection.query(
    query_texts=["my query"],
    include=["documents", "metadatas", "embeddings"],
)