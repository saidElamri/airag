import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.llms import FakeListLLM

# Load environment variables
load_dotenv()

DB_DIR = "./chroma_db"

def get_vectorstore():
    if not os.path.exists(DB_DIR):
        print(f"Error: Database directory {DB_DIR} does not exist. Run ingest.py first.")
        return None
    
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        print("Using Groq LLM (Llama-3)")
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=groq_api_key
        )
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("Using Google Gemini LLM")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=google_api_key
        )
    
    print("Warning: No GOOGLE_API_KEY found. Using Fake LLM for testing.")
    return FakeListLLM(responses=[
        "Ceci est une réponse simulée car aucune clé API n'a été trouvée.",
        "Pour réinitialiser votre mot de passe, allez sur le portail IT.",
        "L'erreur 404 signifie que la ressource est introuvable."
    ])

def get_rag_chain():
    vectorstore = get_vectorstore()
    if not vectorstore:
        return None

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = get_llm()

    # LCEL Implementation
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI

    template = """Vous êtes un expert en support informatique. Utilisez les extraits de contexte suivants pour répondre à la question. 
    Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain # Changed from rag_chain

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Usage: python rag.py <your question>")
        # Default test query
        query = "Comment réinitialiser le mot de passe ?"

    print(f"Question: {query}")
    rag_chain = get_rag_chain()
    
    if rag_chain:
        try:
            result = rag_chain.invoke({"query": query})
            print("\n--- Answer ---")
            print(result["result"])
            print("\n--- Sources ---")
            for doc in result["source_documents"]:
                print(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:100]}...")
        except Exception as e:
            print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
