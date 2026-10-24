# Statistical Text Representation: TF-IDF and Bag-of-Words

The Bag-of-Words model represents text documents as unordered frequency histograms of constituent vocabulary tokens, discarding grammatical syntax and word order. Term Frequency-Inverse Document Frequency (TF-IDF) refines Bag-of-Words by weighting tokens according to their corpus-wide specificity.

Term Frequency (TF) measures how frequently a term appears in a given document: TF(t, d) = count(t, d) / total_words(d). Inverse Document Frequency (IDF) penalizes ubiquitous words (such as 'the' or 'is') that appear across almost all documents: IDF(t) = log(N / (df(t) + 1)). The composite weight TF*IDF identifies distinctive domain keywords, creating high-dimensional sparse document vectors suitable for lexical search and text classification.
