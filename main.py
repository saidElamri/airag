from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import models, auth, database, rag
from database import get_db
import time
import mlflow
import mlflow.pyfunc
import os

# MLflow Setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "postgresql://aiuser:aipassword@localhost:5432/airag_db"))
mlflow.set_experiment("IT_Support_RAG")

# Global flag for model registration
MODEL_REGISTERED = False

class RAGAssistant(mlflow.pyfunc.PythonModel):
    """
    A wrapper class for the RAG pipeline to be logged as an MLflow pyfunc model.
    """
    def load_context(self, context):
        import rag
        self.chain = rag.get_rag_chain()

    def predict(self, context, model_input):
        if isinstance(model_input, list):
            query = model_input[0]
        else:
            query = model_input
        
        result = self.chain.invoke({"query": query})
        return result["result"]


# Create DB tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Assistant RAG Support IT")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Assistant RAG"}

@app.post("/auth/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # In a real scenario, check DB for user. For now, we'll create/verify a demo user or just allow any email with matching pwd hash (simplified for MVP)
    # Actually, let's implement basic user fetch
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    
    if not user:
        # Auto-register for demo purposes if user doesn't exist
        hashed_pw = auth.get_password_hash(form_data.password)
        user = models.User(email=form_data.username, hashedpassword=hashed_pw, isactive=True)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if not user.isactive:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
            )
        if not auth.verify_password(form_data.password, user.hashedpassword):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/query")
async def query_rag(question: str, current_user: str = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    start_time = time.time()
    
    # Get RAG chain
    qa_chain = rag.get_rag_chain()
    if not qa_chain:
        raise HTTPException(status_code=500, detail="RAG system not initialized (Vector DB missing?)")
    
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        
        # Calculate similarity score for compliance (brief req)
        # We manually query Chroma for the top doc score
        vectorstore = rag.get_vectorstore()
        top_docs = vectorstore.similarity_search_with_score(question, k=1)
        similarity_score = 0
        if top_docs:
            # Chroma returns (doc, distance). Similarity can be 1-distance or similar logic
            similarity_score = 1.0 / (1.0 + top_docs[0][1])  # Simplified score
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    latency = (time.time() - start_time) * 1000
    
    try:
        with mlflow.start_run():
            # Log Model to Registry (Once per session)
            global MODEL_REGISTERED
            if not MODEL_REGISTERED:
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=RAGAssistant(),
                    registered_model_name="IT-Support-Assistant",
                    code_paths=["rag.py", "auth.py", "database.py", "models.py"]
                )
                MODEL_REGISTERED = True

            mlflow.log_param("question", question)
            mlflow.log_metric("latency_ms", latency)
            mlflow.log_metric("similarity_score", similarity_score)
            mlflow.log_dict({"answer": answer}, "response.json")
            
            # Log chunks and full prompt
            if "source_documents" in result:
                chunks_info = [
                    {"page": doc.metadata.get("page", "N/A"), "content": doc.page_content[:200]}
                    for doc in result["source_documents"]
                ]
                mlflow.log_dict({"chunks": chunks_info}, "chunks.json")
                
                # Reconstruct and log full prompt for observability
                context_text = "\n\n".join([doc.page_content for doc in result["source_documents"]])
                full_prompt = f"""Vous êtes un expert en support informatique. Utilisez les extraits de contexte suivants pour répondre à la question. 
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

Context: {context_text}
Question: {question}
Answer:"""
                mlflow.log_text(full_prompt, "full_prompt.txt")
                
            # Log LLM Info
            mlflow.log_param("llm_provider", "Groq" if os.getenv("GROQ_API_KEY") else "Google")
            mlflow.log_param("model_name", "llama-3.3-70b-versatile" if os.getenv("GROQ_API_KEY") else "gemini-2.0-flash")
            mlflow.log_param("temperature", 0)
    except Exception as ml_err:
        print(f"MLflow Error: {ml_err}")
    
    # Save query to DB
    user = db.query(models.User).filter(models.User.email == current_user).first()
    if user:
        db_query = models.Query(
            userid=user.id,
            question=question,
            answer=answer,
            latency_ms=latency
        )
        db.add(db_query)
        db.commit()
    
    return {"question": question, "answer": answer, "latency_ms": latency}

@app.get("/history")
async def get_history(current_user: str = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == current_user).first()
    if not user:
        return []
    
    queries = db.query(models.Query).filter(models.Query.userid == user.id).order_by(models.Query.created_at.desc()).all()
    return queries
