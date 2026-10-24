# Dense Word Embeddings: Word2Vec, GloVe, and Semantic Geometry

Dense word embeddings represent vocabulary tokens as continuous, low-dimensional vectors (e.g. 100 to 768 dimensions) where semantically related words occupy proximate coordinates in geometric vector space. Word2Vec learns these representations using shallow neural architectures: Continuous Bag of Words (CBOW) predicts target center words from context windows, while Skip-gram predicts surrounding context words given a center word using negative sampling.

Dense embeddings exhibit linear semantic compositionality, famously demonstrated by vector arithmetic: vector('King') - vector('Man') + vector('Woman') ≈ vector('Queen'). Global Vectors (GloVe) achieves comparable representations by factorizing global log-bilinear word co-occurrence matrices across the entire corpus.
