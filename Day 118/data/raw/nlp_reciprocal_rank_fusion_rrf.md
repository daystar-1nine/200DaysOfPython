# Reciprocal Rank Fusion (RRF) for Hybrid Search

Reciprocal Rank Fusion (RRF) is an unsupervised rank aggregation method that combines multiple ranked result lists into a single consolidated ranking.
The RRF score for document $d$ across ranking systems $R$ is defined as:
$$\operatorname{RRF}(d) = \sum_{r \in R} \frac{1}{k_0 + \operatorname{rank}_r(d)}$$
where $k_0$ is a smoothing constant (conventionally set to 60) and $\operatorname{rank}_r(d)$ is the 1-based position of document $d$ in system $r$.

RRF is parameter-free, requires no score calibration or normalization across disparate scoring distributions (such as unbounded BM25 scores versus bounded cosine similarities), and reliably outperforms either individual retriever.