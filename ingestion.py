# ingestion.py
#
# IMPORTANT: This is a one-time script you run from your LOCAL machine.
# Its purpose is to populate your CLOUD Redis database.
#
# Before running, set these environment variables in your terminal:
# $env:REDIS_HOST = "your-cloud-redis-host.com"
# $env:REDIS_PORT = "12345"
# $env:REDIS_PASSWORD = "your-cloud-redis-password"
#
# Then, run:
# python ingestion.py
# -----------------------------------------------------------------

import os
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime
from redis.commands.search.field import (
    TextField, VectorField, NumericField, TagField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition, IndexType
)

# --- Copied DocumentStore Class ---
# We must include the class definition in this script
# so it can run independently.
class DocumentStore:
    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_password: str
    ):
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=False,
                socket_keepalive=True
            )
            self.redis_client.ping()
            print("DocumentStore: Connected to Redis successfully!")
        except Exception as e:
            print(f"DocumentStore: Error connecting to Redis: {e}")
            raise

        print("DocumentStore: Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384
        self.index_name = "rag_doc_idx"
        self._create_document_index()

    def _create_document_index(self):
        try:
            self.redis_client.ft(self.index_name).info()
            print(f"DocumentStore: Index '{self.index_name}' already exists.")
        except redis.exceptions.ResponseError:
            print(f"DocumentStore: Creating index '{self.index_name}'...")
            schema = [
                TextField("text"),
                VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": self.embedding_dim, "DISTANCE_METRIC": "COSINE"}),
                TagField("domain"),
                TextField("source_url"),
                NumericField("pub_date"),
                NumericField("trust_score"),
                TextField("author")
            ]
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
            )
            print(f"DocumentStore: Created index '{self.index_name}'.")

    def index_document(
        self, 
        text_chunk: str, 
        domain: str, 
        source_url: str, 
        pub_date: datetime,
        trust_score: float = 0.8,
        author: str = "Unknown"
    ):
        try:
            embedding = self.model.encode(text_chunk, convert_to_numpy=True)
            embedding_bytes = embedding.astype(np.float32).tobytes()
            chunk_hash = hashlib.md5(text_chunk.encode()).hexdigest()
            doc_key = f"doc:{domain}:{chunk_hash}"
            doc_data = {
                "text": text_chunk,
                "embedding": embedding_bytes,
                "domain": domain,
                "source_url": source_url,
                "pub_date": pub_date.timestamp(),
                "trust_score": trust_score,
                "author": author
            }
            self.redis_client.hset(doc_key, mapping=doc_data)
            print(f"  > Indexed: {source_url}")
        except Exception as e:
            print(f"DocumentStore: Error indexing document {source_url}: {e}")

# --- Main Ingestion Logic ---
def main():
    print("Starting ingestion process...")
    
    # Load credentials from environment variables
    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT")
    password = os.getenv("REDIS_PASSWORD")

    if not all([host, port, password]):
        print("Error: REDIS_HOST, REDIS_PORT, and REDIS_PASSWORD environment variables must be set.")
        return

    try:
        port_int = int(port)
    except ValueError:
        print("Error: REDIS_PORT must be a number.")
        return

    store = DocumentStore(
        redis_host=host,
        redis_port=port_int,
        redis_password=password
    )

    print("\nIndexing Medical Documents...")
    store.index_document(
        text_chunk="Symptoms of influenza (flu) include fever, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue. The flu is a contagious respiratory illness caused by influenza viruses. Source: CDC.",
        domain="medical",
        source_url="https.www.cdc.gov/flu/symptoms/index.html",
        pub_date=datetime(2024, 9, 15),
        trust_score=1.0,
        author="CDC"
    )
    store.index_document(
        text_chunk="Paracetamol is commonly used to treat pain and fever. It is important to follow the recommended dosage. Source: NHS.",
        domain="medical",
        source_url="https.nhs.uk/medicines/paracetamol-for-adults/",
        pub_date=datetime(2024, 1, 10),
        trust_score=1.0,
        author="NHS"
    )
    # The new, more detailed document
    store.index_document(
        text_chunk="A fever, or high temperature, is a body temperature of 38C (100.4F) or higher. A fever is usually a sign that your body is trying to fight an infection. You can treat a fever yourself with paracetamol.",
        domain="medical",
        source_url="https.nhs.uk/conditions/fever-in-adults/",
        pub_date=datetime(2024, 3, 1),
        trust_score=1.0,
        author="NHS"
    )

    print("\nIndexing Financial Documents...")
    store.index_document(
        text_chunk="Form 10-K is an annual report required by the U.S. Securities and Exchange Commission (SEC), that gives a comprehensive summary of a company's financial performance.",
        domain="finance",
        source_url="https://www.sec.gov/edgar",
        pub_date=datetime(2023, 7, 1),
        trust_score=1.0,
        author="SEC"
    )
    store.index_document(
        text_chunk="The Consumer Duty, set by the FCA, introduces a more assertive and data-led approach to regulation. It requires firms to act to deliver good outcomes for retail customers.",
        domain="finance",
        source_url="https.fca.org.uk/firms/consumer-duty",
        pub_date=datetime(2024, 5, 20),
        trust_score=1.0,
        author="FCA"
    )
    
    print("\nIngestion complete. 5 documents indexed.")

if __name__ == "__main__":
    main()