# Cross-Encoders vs Bi-Encoders in Neural Reranking

Information retrieval systems employ two contrasting neural architectures: Bi-Encoders and Cross-Encoders.
In a Bi-Encoder architecture, the query and document are encoded completely independently into dense vector embeddings $u$ and $v$.
Relevance is scored by computing their dot product or cosine similarity.
Bi-encoders are fast and enable index pre-computation with MIPS/ANN libraries, but sacrifice fine-grained cross-token interactions.

In a Cross-Encoder architecture, query and document are concatenated as a single input sequence `[CLS] query [SEP] document [SEP]` and processed simultaneously through all transformer layers.
Every token in the query attends directly to every token in the document via full cross-attention.
While Cross-Encoders provide substantially superior ranking accuracy (nDCG@5), their $O(N)$ inference cost necessitates deploying them as a second-stage reranker over top candidate pools.