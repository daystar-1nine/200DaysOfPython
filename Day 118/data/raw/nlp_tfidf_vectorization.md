# TF-IDF Weighting and Vector Space Retrieval

Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic reflecting how important a word is to a document in a collection.
The term frequency $TF(t, d)$ measures the raw or sublinear count of term $t$ in document $d$.
The inverse document frequency $IDF(t) = \ln(1 + N / n_t)$ penalizes words that appear ubiquitously across all documents (such as 'the', 'is', 'and').

In the vector space model, documents and queries are represented as sparse high-dimensional vectors with TF-IDF weights.
Relevance is scored by computing the cosine similarity between the query vector and document vectors.
While TF-IDF is computationally efficient and interpretable, it suffers from vocabulary mismatch and cannot capture semantic synonyms.