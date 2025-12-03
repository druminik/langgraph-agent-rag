# build_index.py
import os
import glob

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_FOLDER = "./docs"
INDEX_DIR = "./faiss_index"

# --- Embeddings must match what you use in the agent ---
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 64},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

def build_vectorstore():
    paths = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf")) + \
            glob.glob(os.path.join(DOCS_FOLDER, "*.docx"))

    if not paths:
        print(f"No PDF or DOCX files found in folder: {DOCS_FOLDER}")
        return

    docs = []
    for path in paths:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())

    all_splits = []
    for d in docs:
        splits = text_splitter.split_documents([d])
        all_splits.extend(splits)

    texts = []
    metadatas = []
    for split in all_splits:
        texts.append(split.page_content)
        meta = dict(split.metadata)
        source_path = meta.get("source", "")
        if source_path:
            meta["filename"] = os.path.basename(source_path)
        else:
            meta["filename"] = "unknown"
        metadatas.append(meta)

    if not texts:
        print("Documents found but no text extracted.")
        return

    print(f"Embedding {len(texts)} chunks…")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved FAISS index to {INDEX_DIR}")

if __name__ == "__main__":
    build_vectorstore()
