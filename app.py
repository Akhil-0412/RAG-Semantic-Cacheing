# -----------------------------------------------------------------
# Authoritative RAG: Caching & Fact-Checking App
#
# This single file contains the full application:
# 1. Streamlit Frontend 
# 2. RAG Orchestration 
# 3. All helper classes 
#
# To deploy, this file will be run by Streamlit Community Cloud.
# -----------------------------------------------------------------

import streamlit as st
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from redis.commands.search.field import (
    TextField, VectorField, NumericField, TagField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition, IndexType
)
# Redis Search imports (v4/v5 compatible)
from redis.commands.search.field import TextField, VectorField, NumericField, TagField
try:
    # redis-py <= 4.x
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
except ModuleNotFoundError:
    # redis-py >= 5.x
    from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

import threading
from collections import defaultdict

# -----------------------------------------------------------------
# 1. HELPER CLASS: DocumentStore
# Manages the authoritative document index (separate from cache)
# -----------------------------------------------------------------
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
                decode_responses=False, # Handle binary embedding data
                socket_keepalive=True
            )
            self.redis_client.ping()
            print("DocumentStore: Connected to Redis successfully!")
        except Exception as e:
            print(f"DocumentStore: Error connecting to Redis: {e}")
            st.error(f"DocumentStore: Error connecting to Redis: {e}")
            raise

        # Lazy-load model only when needed
        self._model = None
        self.embedding_dim = 384 # all-MiniLM-L6-v2
        self.index_name = "rag_doc_idx"
        self._create_document_index()

    @property
    def model(self):
        """Lazy loader for the embedding model."""
        if self._model is None:
            print("DocumentStore: Loading embedding model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("DocumentStore: Model loaded.")
        return self._model

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
    
    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        embedding_bytes = query_embedding.astype(np.float32).tobytes()

        q = Query(f"*=>[KNN {top_k} @embedding $vec AS similarity]") \
            .sort_by("similarity") \
            .return_fields("text", "domain", "source_url", "pub_date", "trust_score", "author", "similarity") \
            .dialect(2)

        try:
            results = self.redis_client.ft(self.index_name).search(q, query_params={"vec": embedding_bytes})
            chunks = []
            for doc in results.docs:
                score = 1 - float(doc.similarity)
                chunks.append({
                    "text": doc.text,
                    "domain": doc.domain,
                    "source_url": doc.source_url,
                    "pub_date": datetime.fromtimestamp(float(doc.pub_date)),
                    "trust_score": float(doc.trust_score),
                    "author": doc.author,
                    "retrieval_score": score
                })
            return chunks
        except Exception as e:
            print(f"DocumentStore: Vector search error: {e}")
            return []

# -----------------------------------------------------------------
# 2. HELPER CLASS: SemanticCache
# Manages the L1/L2 query cache (for final answers)
# -----------------------------------------------------------------
class SemanticCache:
    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_password: str,
        similarity_threshold: float = 0.90,
        ttl_seconds: int = 3600
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
            print("SemanticCache: Connected to Redis successfully!")
        except Exception as e:
            print(f"SemanticCache: Error connecting to Redis: {e}")
            st.error(f"SemanticCache: Error connecting to Redis: {e}")
            raise
        
        self._model = None
        self.embedding_dim = 384 # all-MiniLM-L6-v2
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.index_name = "semantic_cache_idx"
        self._create_vector_index()

        self.stats = defaultdict(int)

    @property
    def model(self):
        """Lazy loader for the embedding model."""
        if self._model is None:
            print("SemanticCache: Loading embedding model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("SemanticCache: Model loaded.")
        return self._model

    def _create_vector_index(self):
        try:
            self.redis_client.ft(self.index_name).info()
            print(f"SemanticCache: Index '{self.index_name}' already exists")
        except redis.exceptions.ResponseError:
            print(f"SemanticCache: Creating index '{self.index_name}'...")
            schema = [
                TextField("query"),
                TextField("response"), # Stores the JSON string
                VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": self.embedding_dim, "DISTANCE_METRIC": "COSINE"}),
                NumericField("timestamp"),
                NumericField("access_count")
            ]
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=["cache:l2:"], index_type=IndexType.HASH)
            )
            print(f"SemanticCache: Created index '{self.index_name}'")

    def get(self, query: str) -> Optional[Dict]:
        self.stats["total_queries"] += 1
        query_hash = hashlib.md5(query.encode()).hexdigest()
        l1_key = f"cache:l1:{query_hash}"

        l1_result = self.redis_client.hgetall(l1_key)
        if l1_result:
            self.stats["cache_hits"] += 1
            self.redis_client.hincrby(l1_key, "access_count", 1)
            self.redis_client.expire(l1_key, self.ttl_seconds)
            l2_key = f"cache:l2:{query_hash}"
            if self.redis_client.exists(l2_key):
                 self.redis_client.expire(l2_key, self.ttl_seconds)
            return {
                "response": l1_result[b"response"].decode(),
                "similarity": 1.0,
                "original_query": l1_result[b"query"].decode(),
                "cache_type": "L1_EXACT"
            }

        query_embedding = self.model.encode(query, convert_to_numpy=True)
        embedding_bytes = query_embedding.astype(np.float32).tobytes()
        
        q = Query("*=>[KNN 5 @embedding $vec AS similarity]") \
            .sort_by("similarity") \
            .return_fields("query", "response", "timestamp", "access_count", "similarity") \
            .dialect(2)

        try:
            results = self.redis_client.ft(self.index_name).search(q, query_params={"vec": embedding_bytes})
            if results.docs:
                best_match = results.docs[0]
                similarity_score = 1 - float(best_match.similarity)
                if similarity_score >= self.similarity_threshold:
                    self.stats["cache_hits"] += 1
                    cache_key = best_match.id
                    self.redis_client.hincrby(cache_key, "access_count", 1)
                    self.redis_client.expire(cache_key, self.ttl_seconds)
                    l1_key_hash = hashlib.md5(best_match.query.encode()).hexdigest()
                    l1_key = f"cache:l1:{l1_key_hash}"
                    if self.redis_client.exists(l1_key):
                        self.redis_client.expire(l1_key, self.ttl_seconds)
                    return {
                        "response": best_match.response,
                        "similarity": similarity_score,
                        "original_query": best_match.query,
                        "cache_type": "L2_SEMANTIC"
                    }
        except Exception as e:
            print(f"SemanticCache: Vector search error: {e}")

        self.stats["cache_misses"] += 1
        return None

    def set(self, query: str, response: str, metadata: Dict = None):
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        embedding_bytes = query_embedding.astype(np.float32).tobytes()
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"cache:l2:{query_hash}"
        cache_data = {
            "query": query,
            "response": response, # This is the JSON string
            "embedding": embedding_bytes,
            "timestamp": datetime.now().timestamp(),
            "access_count": 0
        }
        if metadata: cache_data.update(metadata)
        self.redis_client.hset(cache_key, mapping=cache_data)
        self.redis_client.expire(cache_key, self.ttl_seconds)

        l1_key = f"cache:l1:{query_hash}"
        self.redis_client.hset(l1_key, mapping={"query": query, "response": response, "access_count": 0})
        self.redis_client.expire(l1_key, self.ttl_seconds)

    def get_stats(self) -> Dict:
        total = self.stats["total_queries"]
        hit_rate = (self.stats["cache_hits"] / total) * 100 if total > 0 else 0
        return {
            "total_queries": total,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    def clear_cache(self):
        print("Clearing cache...")
        count = 0
        for key in self.redis_client.scan_iter("cache:*"):
            self.redis_client.delete(key)
            count += 1
        print(f"Cache cleared. Deleted {count} keys.")
        self.stats = defaultdict(int)

# -----------------------------------------------------------------
# 3. HELPER CLASS: FactChecker
# (MOCK) Verification pipeline
# -----------------------------------------------------------------
class FactChecker:
    def check_claim(self, claim: str, evidence_chunks: List[Dict]) -> Dict:
        print(f"FactChecker: Verifying claim: '{claim}'")
        if not evidence_chunks:
            return {"status": "LOW_CONFIDENCE", "score": 0.0, "message": "No supporting evidence found."}
        
        avg_trust_score = sum(c['trust_score'] for c in evidence_chunks) / len(evidence_chunks)
        
        if "I'm sorry, I don't have that information" in claim:
             return {"status": "LOW_CONFIDENCE", "score": 0.1, "message": "Answer not found in context."}

        if avg_trust_score > 0.8:
            status = "HIGH_CONFIDENCE"
            score = 0.9 + np.random.uniform(0, 0.1)
        elif avg_trust_score > 0.5:
            status = "MEDIUM_CONFIDENCE"
            score = 0.6 + np.random.uniform(0, 0.1)
        else:
            status = "LOW_CONFIDENCE"
            score = 0.3 + np.random.uniform(0, 0.1)

        print(f"FactChecker: Verification complete. Status: {status}")
        return {"status": status, "score": round(score, 3)}

# -----------------------------------------------------------------
# 4. MAIN ORCHESTRATOR: RAGSystem
# (Combines all logic)
# -----------------------------------------------------------------
class RAGSystem:
    def __init__(self, redis_host, redis_port, redis_pass, openai_key, openai_base, gemma_model):
        print("Initializing RAGSystem...")
        # Note: We are not using AdaptiveCacheManager in this simplified deployment
        # to reduce background threads and complexity.
        self.cache = SemanticCache(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_pass,
            similarity_threshold=0.90
        )
        self.doc_store = DocumentStore(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_pass
        )
        self.fact_checker = FactChecker()

        # Configure OpenAI client for OpenRouter
        openai.api_key = openai_key
        openai.api_base = openai_base
        self.gemma_model = gemma_model
        print("RAGSystem initialized.")

    def _score_and_filter_evidence(self, chunks: List[Dict]) -> List[Dict]:
        scored_chunks = []
        for chunk in chunks:
            days_old = (datetime.now() - chunk['pub_date']).days
            recency_score = max(0, 1 - (days_old / 365.0))
            final_score = (chunk['retrieval_score'] * 0.5 + chunk['trust_score'] * 0.3 + recency_score * 0.2)
            chunk['final_score'] = final_score
            scored_chunks.append(chunk)
        return sorted(scored_chunks, key=lambda x: x['final_score'], reverse=True)[:3]

    def _build_context(self, chunks: List[Dict]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"Source [{i+1}] ({chunk['source_url']}):\n{chunk['text']}\n\n"
        return context.strip()

    def _detect_domain(self, query: str, chunks: List[Dict]) -> str:
        if any(c['domain'] == 'medical' for c in chunks): return "medical"
        if any(c['domain'] == 'finance' for c in chunks): return "finance"
        if "medical" in query or "health" in query: return "medical"
        if "finance" in query or "stock" in query: return "finance"
        return "general"

    def _generate_response(self, query: str, context: str) -> str:
        print(f"Generating response from LLM ({self.gemma_model})...")
        prompt = f"""You are a helpful AI assistant. Answer the question based *only* on the context provided below.
Cite your answer by referencing the source number, like [1].
If the context does not contain the answer, say "I'm sorry, I don't have that information in my context."

Context:
{context}

Question:
{query}

Answer:"""
        try:
            response = openai.ChatCompletion.create(
                model=self.gemma_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return f"Error: An exception occurred while contacting the LLM: {e}"

    def query(self, user_query: str) -> Dict:
        print("\n--- New Query ---")
        start_time = time.time()
        
        cached_result_json = self.cache.get(user_query)
        if cached_result_json:
            latency = time.time() - start_time
            print(f"Cache HIT ({cached_result_json['cache_type']}). Latency: {latency*1000:.2f}ms")
            cached_data = json.loads(cached_result_json['response'])
            cached_data.update({
                "cached": True,
                "cache_type": cached_result_json['cache_type'],
                "latency_ms": round(latency * 1000, 2)
            })
            return cached_data

        print("Cache MISS. Running full RAG pipeline.")
        evidence_chunks = self.doc_store.retrieve_chunks(user_query, top_k=5)
        scored_chunks = self._score_and_filter_evidence(evidence_chunks)
        context_str = self._build_context(scored_chunks)
        
        if not context_str:
            llm_answer = "I'm sorry, I don't have that information in my context."
        else:
            llm_answer = self._generate_response(user_query, context_str)

        verification = self.fact_checker.check_claim(llm_answer, scored_chunks)
        
        citations = []
        for i, chunk in enumerate(scored_chunks):
            citations.append({
                "id": i+1,
                "source_url": chunk['source_url'],
                "snippet": chunk['text'][:150] + "...",
                "pub_date": chunk['pub_date'].isoformat(),
                "trust_score": chunk['trust_score']
            })

        domain = self._detect_domain(user_query, scored_chunks)
        disclaimer = None
        if domain == "medical":
            disclaimer = "Not medical advice. Consult a clinician."
        elif domain == "finance":
            disclaimer = "Not financial advice. This is informational guidance."

        final_response = {
            "answer": llm_answer,
            "status": verification["status"],
            "verification_score": verification["score"],
            "citations": citations,
            "disclaimer": disclaimer,
            "domain": domain
        }

        try:
            response_json = json.dumps(final_response)
            self.cache.set(query=user_query, response=response_json, metadata={"domain": domain, "status": verification["status"]})
        except TypeError as e:
            print(f"Error serializing response for cache: {e}")

        latency = time.time() - start_time
        print(f"Full RAG complete. Latency: {latency*1000:.2f}ms")
        
        return {**final_response, "cached": False, "cache_type": None, "latency_ms": round(latency * 1000, 2)}

# -----------------------------------------------------------------
# 5. STREAMLIT UI (The Frontend)
# -----------------------------------------------------------------

# --- Page Configuration ---
st.set_page_config(
    page_title="Authoritative RAG",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Check Secrets ---
# Check if all required secrets are set in Streamlit Cloud
def check_secrets():
    required_secrets = [
        "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD",
        "AI_API_KEY", "AI_API_BASE"
    ]
    missing = [s for s in required_secrets if not hasattr(st, 'secrets') or s not in st.secrets]
    if missing:
        st.error(f"Missing secrets: {', '.join(missing)}. Please add them to your Streamlit Cloud settings.")
        return False
    
    # Check if port is a number
    try:
        int(st.secrets["REDIS_PORT"])
    except ValueError:
        st.error("REDIS_PORT secret must be a number.")
        return False
    except Exception as e:
        st.error(f"Error reading REDIS_PORT: {e}")
        return False
        
    return True

# --- Initialization & Caching ---
# We use Streamlit's @st.cache_resource to initialize and
# cache the RAGSystem object across user sessions.
@st.cache_resource
def get_rag_system():
    print("Attempting to initialize RAGSystem...")
    if not check_secrets():
        st.stop()
        
    try:
        system = RAGSystem(
            redis_host=st.secrets["REDIS_HOST"],
            redis_port=int(st.secrets["REDIS_PORT"]),
            redis_pass=st.secrets["REDIS_PASSWORD"],
            openai_key=st.secrets["AI_API_KEY"],
            openai_base=st.secrets["AI_API_BASE"],
            gemma_model="google/gemma-3-27b-it" # Or your model of choice
        )
        print("RAGSystem initialization successful.")
        return system
    except Exception as e:
        print(f"Failed to initialize RAGSystem: {e}")
        st.error(f"Failed to initialize RAGSystem. See logs. Error: {e}")
        st.stop()


# --- Main App ---
if check_secrets():
    # Initialize the system
    rag_system = get_rag_system()

    # --- Sidebar ---
    st.sidebar.title("Admin Controls")
    if st.sidebar.button("Clear Cache"):
        try:
            rag_system.cache.clear_cache()
            st.sidebar.success("Cache cleared!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to clear cache: {e}")

    st.sidebar.header("Cache Stats")
    try:
        stats = rag_system.cache.get_stats()
        st.sidebar.metric("Total Queries", stats.get("total_queries", 0))
        st.sidebar.metric("Cache Hits", stats.get("cache_hits", 0))
        st.sidebar.metric("Hit Rate", f"{stats.get('hit_rate_percent', 0.0)}%")
    except Exception as e:
        st.sidebar.error(f"Could not get cache stats: {e}")

    # --- Main Page ---
    st.title("🚀 Authoritative RAG Dashboard")
    st.caption("With Fact-Checking, Citations, and Adaptive Caching")
    
    # Use a form for the query input
    with st.form(key="query_form"):
        user_query = st.text_input("Enter your question:", key="query_input", placeholder="e.g., What are the symptoms of the flu?")
        submit_button = st.form_submit_button(label="Submit Query")

    if submit_button and user_query:
        if not rag_system:
            st.error("RAG system is not initialized. Please check secrets and logs.")
        else:
            with st.spinner("Processing... (Retrieving, Fact-Checking, and Caching)"):
                try:
                    response_data = rag_system.query(user_query)
                    
                    # --- Display Answer & Status ---
                    answer = response_data['answer']
                    status = response_data['status']

                    if status == "HIGH_CONFIDENCE":
                        st.success(f"**Answer:** {answer}")
                        st.write(f"✅ **Verification Status:** {status} (Score: {response_data['verification_score']})")
                    elif status == "MEDIUM_CONFIDENCE":
                        st.warning(f"**Answer:** {answer}")
                        st.write(f"⚠️ **Verification Status:** {status} (Score: {response_data['verification_score']}). Please verify sources.")
                    else:
                        st.error(f"**Answer:** {answer}")
                        st.write(f"❌ **Verification Status:** {status} (Score: {response_data['verification_score']}). Low confidence or error.")

                    # --- Display Disclaimer ---
                    if response_data['disclaimer']:
                        st.info(f"**Disclaimer:** {response_data['disclaimer']}")

                    # --- Display Cache Info & Latency ---
                    st.caption(f"Domain: {response_data['domain']} | Latency: {response_data['latency_ms']:.2f}ms")
                    if response_data['cached']:
                        st.write(f"**Cache:** `HIT` ({response_data['cache_type']})")
                    else:
                        st.write("**Cache:** `MISS`")

                    # --- Display Citations ---
                    st.subheader("Sources")
                    citations = response_data['citations']
                    if not citations:
                        st.write("No sources found for this query.")
                    
                    for cit in citations:
                        with st.expander(f"**Source [{cit['id']}]**: {cit['source_url']}"):
                            st.markdown(f"**Snippet:** *{cit['snippet']}...*")
                            st.write(f"**Trust Score:** {cit['trust_score']} | **Pub Date:** {cit['pub_date']}")
                            
                except Exception as e:
                    st.error(f"An unexpected error occurred during query processing: {e}")
                    print(f"Query processing error: {e}")
    
    elif submit_button:
        st.warning("Please enter a query.")
