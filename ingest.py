import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
import glob

# Configuration
PDF_DIR = "./pdf_data"
DB_DIR = "./chroma_db"
# FastEmbed uses "BAAI/bge-small-en-v1.5" by default or others, but handles download internally suitable for CPU
 

def main():
    print("Starting ingestion process...")
    
    # 1. Locate PDF
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    pdf_path = pdf_files[0]
    print(f"Processing file: {pdf_path}")

    # 2. Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # 4. Generate Embeddings & Store in ChromaDB
    print("Generating embeddings and creating vector store...")
    # Using HuggingFace (sentence-transformers)
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"Vector store created and persisted at {DB_DIR}")

if __name__ == "__main__":
    main()
