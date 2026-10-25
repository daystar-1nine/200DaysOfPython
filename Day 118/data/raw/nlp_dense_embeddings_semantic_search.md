# Dense Embeddings and Semantic Vector Retrieval

Dense vector representations map textual sequences into continuous low-dimensional geometric spaces (typically 64 to 1536 dimensions) where semantic similarity corresponds to vector proximity.
Models encode latent meaning beyond exact token matches, allowing queries like 'feline health' to match documents about 'cat medical care'.

Nearest neighbor search in dense vector spaces relies on cosine similarity:
$$\text{Sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
For web-scale search over billions of vectors, Approximate Nearest Neighbor (ANN) indexes such as HNSW (Hierarchical Navigable Small World) and IVF-PQ provide logarithmic query latency.