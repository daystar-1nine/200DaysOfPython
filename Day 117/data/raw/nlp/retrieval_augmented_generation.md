# Retrieval-Augmented Generation (RAG) Architecture and Grounding

Retrieval-Augmented Generation (RAG) combines neural information retrieval with autoregressive language generation. Rather than relying solely on parametric knowledge memorized in neural network weights during pretraining, a RAG system dynamically retrieves non-parametric factual chunks from an external indexed document store based on the user query.

The retrieved chunks are concatenated into a structured grounding prompt supplied to the language model. This decoupled design mitigates hallucinations, circumvents training knowledge cutoff limitations, protects private internal organizational data, and enables verifiable source attribution through explicit citation tags.
