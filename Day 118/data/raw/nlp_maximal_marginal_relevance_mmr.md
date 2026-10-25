# Maximal Marginal Relevance (MMR) for Result Diversity

In retrieval systems, returning the top-K most similar documents often leads to extreme informational redundancy, where multiple chunks reiterate identical sentences.
Maximal Marginal Relevance (MMR) resolves this by balancing query relevance against novelty relative to already selected results:
$$\operatorname{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \operatorname{Sim}_1(d_i, Q) - (1 - \lambda) \max_{d_j \in S} \operatorname{Sim}_2(d_i, d_j) \right]$$

Setting $\lambda = 1.0$ prioritizes pure relevance (standard top-K ranking), whereas $\lambda = 0.0$ promotes maximal diversity.
A balanced setting of $\lambda = 0.7$ provides strong query relevance while suppressing duplicate content in LLM context windows.