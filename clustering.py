from sklearn.cluster import KMeans
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import os
from sqlalchemy.orm import Session
from database import SessionLocal
import models
import mlflow
import mlflow.sklearn

# Initialize embeddings (same as ingestion)
# We use the same model to ensure vector space compatibility
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def cluster_queries(queries, n_clusters=5):
    """
    Groups user queries into clusters using KMeans.
    
    Args:
        queries (list of str): List of question texts.
        n_clusters (int): Number of clusters to find.
        
    Returns:
        tuple (list of int, KMeans): Cluster labels and the fitted model.
    """
    if not queries:
        return [], None
        
    # Generate embeddings
    print("Generating embeddings for clustering...")
    vectors = embeddings.embed_documents(queries)
    X = np.array(vectors)
    
    # Perform Clustering
    print(f"Clustering {len(queries)} queries into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    return kmeans.labels_.tolist(), kmeans

def update_clusters_in_db(batch_size=10, n_clusters=3):
    """
    Fetches queries from the DB, clusters them, and saves labels back.
    Also tracks the process in MLflow.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("IT_Support_RAG_Analysis")
    
    db: Session = SessionLocal()
    try:
        # Fetch all queries
        query_records = db.query(models.Query).all()
        if not query_records:
            print("No queries found in DB to cluster.")
            return
            
        texts = [q.question for q in query_records]
        
        with mlflow.start_run(run_name="Query_Clustering"):
            labels, model = cluster_queries(texts, n_clusters=n_clusters)
            
            # Log to MLflow
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_metric("total_queries", len(texts))
            mlflow.sklearn.log_model(model, "kmeans_model")
            print("Clustering tracked in MLflow.")

            # Update records
            for record, label in zip(query_records, labels):
                record.cluster = label
            
            db.commit()
            print(f"Successfully updated {len(query_records)} records with cluster labels.")
    except Exception as e:
        print(f"Error during DB clustering update: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # Synchronize clusters with DB
    update_clusters_in_db(n_clusters=3)

