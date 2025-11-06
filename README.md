🚀 Authoritative RAG: A Fact-Checked, Caching RAG System

This project implements a production-grade, authoritative Retrieval-Augmented Generation (RAG) system with a focus on factuality, compliance, and performance.

It is built in Python using Streamlit, Redis (for vector search and caching), and an external LLM API (like OpenRouter).

Features

Authoritative RAG: The system answers questions only based on a curated set of ingested documents (e.g., CDC, SEC).

Fact-Checking: A mock verification step scores the LLM's claim against the retrieved evidence before showing it to the user.

Multi-Domain Safety: Automatically detects "medical" or "financial" queries and applies appropriate disclaimers.

Source Citations: All answers are returned with citations, including snippets, source URLs, and trust scores.

L1/L2 Semantic Caching: A dual-layer cache in Redis dramatically speeds up responses (from ~2500ms to ~30ms) and saves API costs by caching the full, verified response.

Modern UI: A clean, modern frontend built with Streamlit.

Architecture

This application is consolidated into a single app.py file for easy "headless" deployment on platforms like Streamlit Community Cloud.

UI (Streamlit): The user submits a query.

Cache Check (Redis): The system checks its SemanticCache for a similar query.

IF HIT: Returns the cached (and already verified) response.

IF MISS: Proceeds to the full RAG pipeline.

Retrieve (Redis): The DocumentStore searches its vector index for relevant, authoritative chunks.

Score: Chunks are scored based on trust, recency, and relevance.

Generate (LLM): The top chunks and query are sent to an LLM (e.g., Gemma) via the OpenRouter API for a draft answer.

Verify: The draft answer is fact-checked against the evidence.

Cache (Redis): The final, verified response (with citations) is stored in the cache.

Respond (Streamlit): The final response is shown to the user.

Deployment

This app is designed to be deployed for free on Streamlit Community Cloud, using a free Redis database (e.g., Redis Cloud) for storage and caching.

See the deployment guide in the official documentation.