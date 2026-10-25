# BM25 Probabilistic Ranking Function Architecture

Okapi BM25 is a non-linear ranking function derived from probabilistic relevance theory that improves upon traditional TF-IDF.
Its scoring function is expressed as:
$$\operatorname{BM25}(D, Q) = \sum_{q_i \in Q} \operatorname{IDF}(q_i) \cdot \frac{f(q_i, D)(k_1 + 1)}{f(q_i, D) + k_1(1 - b + b \frac{|D|}{\text{avgdl}})}$$

The parameter $k_1$ controls term frequency saturation; as term frequency increases, its marginal relevance score approaches an asymptotic ceiling.
The parameter $b$ controls document length normalization; when $b = 1$, long documents are heavily penalized for verbosity, while $b = 0$ disables length compensation entirely.
BM25 excels at exact keyword matching, error codes, and technical identifiers.