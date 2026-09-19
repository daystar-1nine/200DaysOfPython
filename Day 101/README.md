# 🚀 DAY 101 / 200 — NLP FUNDAMENTALS & TEXT DATA PROCESSING

## 🎯 Overview
Welcome to the second half of the **200 Days of Python + Data Science** journey! Having mastered Python fundamentals, classical machine learning, deep learning, and computer vision (Days 1–100), Day 101 marks our official entry into **Natural Language Processing (NLP)**.

This project delivers a complete, professional **NLP Text Classification Engine** for SMS Spam Detection. It includes from-scratch mathematical implementations of Bag of Words (BoW) and TF-IDF, rigorous data cleaning and tokenization pipelines, multi-model benchmarking (Dummy Baseline vs Logistic Regression vs Multinomial Naive Bayes), extensive error analysis on False Positives and False Negatives, and 12 distinct analytical visualizations.

---

## 📊 Progress Tracker
```text
DAY 101 / 200
Progress: [████████████████████░░░░░░░░░░░░]
Completed: 101 / 200 Days (50.5% COMPLETE)
Remaining: 99 Days
```

---

## 🧠 Learning Objectives
- Master NLP core concepts: documents, sentences, tokens, vocabulary.
- Implement text normalization: lowercasing, whitespace stripping, URL/email removal, punctuation handling.
- Understand the role and trade-offs of stopwords, stemming, and lemmatization.
- Generate unigrams, bigrams, and general N-grams.
- Build **Bag of Words (BoW) from scratch** without scikit-learn and compare with `CountVectorizer`.
- Formulate and code **TF-IDF from scratch** with L2 normalization and smoothed IDF.
- Prevent NLP data leakage by using strict scikit-learn `Pipeline` objects fitted only on training data.
- Analyze binary classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision.
- Perform in-depth Error Analysis: diagnosing False Positives (legitimate messages blocked) and False Negatives (spam missed).

---

## 🤖 NLP Concepts Explained

### 1. Text as Data
Machines cannot directly calculate mathematical operations on text strings. Every text processing pipeline must transform raw text into a numerical vector representation:
$$\text{Raw Text} \longrightarrow \text{Cleaning} \longrightarrow \text{Tokenization} \longrightarrow \text{Feature Extraction} \longrightarrow \text{Numerical Vector} \longrightarrow \text{Model} \longrightarrow \text{Prediction}$$

### 2. Tokenization & Normalization
- **Tokenization**: Splitting text sequences into individual semantic units (words, subwords, or characters).
- **Normalization**: Standardizing text representations by lowercasing, stripping excessive whitespace, and removing non-informative noise (such as extraneous punctuation or URLs).

### 3. Stopwords: Use with Caution
Common words (e.g., *the, is, at, which*) frequently appear across all documents and may add noise. However, in tasks like sentiment analysis, removing words like *"not"* or *"never"* completely reverses the semantic meaning (e.g., *"This film is not good"* $\to$ *"film good"*). Preprocessing must always be tailored to the specific downstream task.

### 4. Stemming vs Lemmatization
- **Stemming**: Heuristic rule-based truncation of word suffixes (e.g., Porter/Snowball stemmers). Fast but produces non-dictionary stems (e.g., *troubling* $\to$ *troubl*).
- **Lemmatization**: Linguistically informed morphological analysis utilizing vocabulary and part-of-speech tags to return the base dictionary lemma (e.g., *better* $\to$ *good*, *running* $\to$ *run*).

### 5. N-Grams
An n-gram is a contiguous sequence of $n$ items from a given sample of text:
- **Unigram ($n=1$)**: `['machine', 'learning']`
- **Bigram ($n=2$)**: `[('machine', 'learning')]`
N-grams enable the model to capture local context and phrase structure that unigrams lose.

### 6. Bag of Words (BoW)
Represents a document as a multiset of its words, disregarding grammar and word order but keeping frequency.

### 7. TF-IDF (Term Frequency — Inverse Document Frequency)
$$\text{TF}(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in } d}$$
$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + df(t)}\right) + 1$$
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
TF-IDF penalizes words that appear ubiquitously across all documents, boosting unique, highly discriminative keywords.

---

## 📁 Project Structure
```text
Day 101/
├── app/
│   ├── __init__.py
│   ├── main.py                      # Main pipeline execution entry point
│   ├── config.py                    # Project paths & hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate_dataset.py      # Deterministic SMS dataset generator
│   │   ├── loader.py                # Data loading module
│   │   ├── cleaner.py               # Text cleaning & validation
│   │   └── validator.py             # Diagnostic validation & class balance
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── normalization.py         # Lowercasing, punctuation, URLs
│   │   ├── tokenization.py          # Word tokenization
│   │   ├── ngrams.py                # N-gram generation
│   │   └── vocabulary.py            # Vocabulary indexing & word frequency
│   ├── features/
│   │   ├── __init__.py
│   │   ├── bow.py                   # Bag of Words from scratch
│   │   └── tfidf.py                 # TF-IDF from scratch & scikit-learn wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py              # DummyClassifier baseline
│   │   ├── logistic.py              # Logistic Regression classifier
│   │   └── naive_bayes.py           # Multinomial Naive Bayes classifier
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Classification metrics
│   │   ├── confusion_matrix.py      # Confusion matrix breakdown
│   │   └── curves.py                # ROC & Precision-Recall curves
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── error_analysis.py        # False Positive / False Negative diagnostics
│   │   └── insights.py              # Top feature coefficients by class
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── charts.py                # Generates all 12 analytical plots
│   └── report.py                    # Generates output/report.md
├── experiments/
│   ├── __init__.py
│   ├── bow_experiment.py            # BoW Unigrams vs Uni+Bigrams
│   ├── tfidf_experiment.py          # TF-IDF N-grams and min_df variations
│   └── model_comparison.py          # Logistic C vs Naive Bayes alpha tuning
├── coding_challenges/
│   ├── __init__.py
│   ├── challenge_101_text_cleaning.py
│   ├── challenge_101_ngrams.py
│   ├── challenge_101_bow_scratch.py
│   └── challenges_1_to_10.py        # All 10 requested coding challenges
├── data/
│   ├── raw/                         # Raw SMS spam dataset
│   └── processed/                   # Cleaned SMS dataset with metadata
├── output/
│   ├── charts/                      # 12 analytical PNG charts
│   ├── predictions.csv              # Test set predictions with probabilities
│   ├── metrics.csv                  # Model benchmark comparison table
│   └── report.md                    # Generated performance & error report
├── tests/
│   ├── __init__.py
│   ├── test_cleaner.py              # Tests for normalization & data cleaning
│   ├── test_tokenizer.py            # Tests for tokenization
│   ├── test_ngrams.py               # Tests for N-gram generator
│   ├── test_bow.py                  # Tests for from-scratch Bag of Words
│   ├── test_tfidf.py                # Tests for TF-IDF implementations
│   ├── test_models.py               # Tests for model training & pipelines
│   ├── test_evaluation.py           # Tests for metrics & curves
│   └── test_error_analysis.py       # Tests for error extraction & diagnostics
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Measured Benchmark Results

The pipeline was executed deterministically on the SMS Spam dataset (800 samples, 25% test split, stratified):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Average Precision |
|---|---|---|---|---|---|---|
| **Dummy Baseline** | 0.8200 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1800 |
| **Multinomial Naive Bayes** | 0.9850 | 1.0000 | 0.9167 | 0.9565 | 0.9991 | 0.9965 |
| **Logistic Regression (TF-IDF 1-2 Grams)** | **0.9950** | **1.0000** | **0.9722** | **0.9859** | **0.9998** | **0.9993** |

### Confusion Matrix Breakdown (Logistic Regression)
- **True Negatives (Ham correctly preserved)**: 164
- **False Positives (Ham incorrectly blocked)**: 0 (0.0% False Positive Rate!)
- **False Negatives (Spam missed)**: 1
- **True Positives (Spam correctly caught)**: 35

> **Engineering Note**: Achieving **0 False Positives** is paramount in production spam filtering. Missing a rare spam message (False Negative) is an annoyance; flagging a crucial bank notice or personal message as spam (False Positive) is unacceptable.

---

## 📈 12 Analytical Visualizations
All charts are automatically generated and saved in `output/charts/`:
1. `1_class_distribution.png`: Imbalance comparison between ham (82%) and spam (18%).
2. `2_message_length_distribution.png`: Character length comparison (spam messages skew significantly longer due to disclaimers/URLs).
3. `3_word_count_distribution.png`: Token distribution across classes.
4. `4_top_words_overall.png`: Most frequent vocabulary terms.
5. `5_top_spam_words.png`: Top positive Logistic Regression feature weights (`call`, `free`, `claim`, `win`, `prize`).
6. `6_top_ham_words.png`: Top negative Logistic Regression feature weights (`meeting`, `lunch`, `tomorrow`, `thanks`).
7. `7_confusion_matrix.png`: Heatmap of classification outcomes.
8. `8_roc_curve.png`: ROC curve demonstrating near-perfect discriminative capability (AUC = 0.9998).
9. `9_precision_recall_curve.png`: Precision-Recall curve maintaining high precision across high recall thresholds.
10. `10_model_comparison.png`: Bar plot comparing Baseline, Naive Bayes, and Logistic Regression.
11. `11_prediction_confidence_distribution.png`: Bimodal confidence distribution showing clear separation.
12. `12_misclassified_analysis.png`: Breakdown of misclassified vs correctly classified samples.

---

## 🎤 Interview Questions & Answers

### Q1: Why can't traditional ML algorithms directly consume raw text?
**Answer**: Traditional ML algorithms rely on linear algebra, matrix multiplications, and Euclidean/geometric distance metrics. They require numerical feature vectors of fixed dimension. Raw text is unstructured, variable in length, and non-numerical.

### Q2: What is tokenization?
**Answer**: Tokenization is the process of segmenting a sequence of characters into meaningful discrete units called tokens (words, punctuation, subwords, or characters) that form the building blocks for vocabulary mapping.

### Q3: What is a vocabulary?
**Answer**: A vocabulary is a finite, ordered set of unique tokens discovered during the training phase. It maps each unique token to a unique integer index in the feature vector.

### Q4: What are stopwords?
**Answer**: Stopwords are extremely high-frequency grammatical words (e.g., *a, the, in, of*) that carry little task-specific discriminative information in certain tasks like topical classification, but can be crucial in sentiment or grammatical parsing.

### Q5: Stemming vs lemmatization?
**Answer**: 
- *Stemming* applies crude, heuristic chopping of word affixes (fast, but often yields non-words like *studi* for *studying*).
- *Lemmatization* utilizes full morphological dictionaries and part-of-speech context to reduce words to their genuine dictionary lemma (e.g., *saw* $\to$ *see* or *saw* depending on verb vs noun).

### Q6: What is an n-gram?
**Answer**: An n-gram is a contiguous sequence of $n$ tokens from a text. Unigrams are single words ($n=1$), bigrams are two-word pairs ($n=2$), and trigrams are three-word triplets ($n=3$). They capture local phrase context (e.g., *"not happy"* vs *"happy"*).

### Q7: What is Bag of Words?
**Answer**: Bag of Words is a text representation model where a document is represented as a vector of word counts across the global vocabulary, completely disregarding word order and grammar.

### Q8: What information does Bag of Words lose?
**Answer**: BoW loses:
1. **Word order & syntax**: *"Dog bites man"* and *"Man bites dog"* produce identical BoW vectors.
2. **Semantic context**: Negation like *"not good"* is split into independent counts of *"not"* and *"good"*.
3. **Word similarity**: Synonyms like *"car"* and *"automobile"* are treated as orthogonal, unrelated features.

### Q9: What is TF-IDF?
**Answer**: Term Frequency — Inverse Document Frequency evaluates how important a word is to a document relative to a corpus. It multiplies the frequency of a word in a document (TF) by the inverse document frequency (IDF) across all documents, penalizing corpus-wide common words.

### Q10: Why does TF-IDF reduce the importance of words appearing in many documents?
**Answer**: If a word appears in almost every document (e.g., *"email"* in an email dataset), its document frequency $df(t) \approx N$, making $\log(N / df(t)) \approx 0$. Such words provide zero discriminative power for separating classes.

### Q11: Why are text feature matrices often sparse?
**Answer**: A vocabulary may contain 10,000 to 100,000 unique words, but an individual tweet or SMS message typically contains only 10 to 30 words. Consequently, over 99.9% of the values in any document vector are zeroes, requiring sparse matrix storage (e.g., CSR format).

### Q12: Why can Naive Bayes work surprisingly well for text classification?
**Answer**: Despite the "naive" assumption of conditional independence between words given the class, Naive Bayes only needs the *relative rank* of class posterior probabilities to be correct to make the right classification decision, even if the absolute probabilities are poorly calibrated. In high-dimensional sparse text spaces with strong keyword signals (like spam triggers), this holds remarkably well.

### Q13: Why can accuracy be misleading for spam detection?
**Answer**: Due to severe class imbalance (e.g., 98% ham, 2% spam), a naive dummy model that always predicts "ham" achieves 98% accuracy while catching zero spam. Furthermore, accuracy treats a False Positive (blocking an urgent email) with the same penalty as a False Negative (letting a spam message through). Precision, Recall, and F1 provide far more honest evaluations.

### Q14: What is data leakage in NLP?
**Answer**: Data leakage in NLP occurs when information from the test/validation set leaks into the training pipeline. Examples include:
- Fitting the tokenizer, vocabulary, or TF-IDF vectorizer on the full dataset before splitting.
- Computing global document frequencies or scaling parameters using test documents.
- Performing feature selection on the entire corpus.

### Q15: Why should TF-IDF be fitted only on training data?
**Answer**: IDF depends on the document frequencies across the corpus ($N$ and $df(t)$). If fitted on test data, the model gains prior knowledge about word distributions in unseen data that it would never possess in a real production environment. In scikit-learn, this is strictly enforced by calling `fit_transform` on `X_train` and only `transform` on `X_test`.

---

## 💻 Running the Project & Tests

### Installation
```bash
pip install -r requirements.txt
```

### Run the Main Pipeline
```bash
python app/main.py
```

### Run the Experiments
```bash
python experiments/bow_experiment.py
python experiments/tfidf_experiment.py
python experiments/model_comparison.py
```

### Run the Coding Challenges
```bash
python coding_challenges/challenges_1_to_10.py
python coding_challenges/challenge_101_bow_scratch.py
```

### Run the Test Suite (40+ Tests)
```bash
pytest -v
```

---

## 🏆 Key Learnings
1. **Text Preprocessing is Task-Dependent**: Aggressive stripping of stopwords or numbers helps some topic classification tasks, but destroys essential information in spam detection and sentiment analysis.
2. **From-Scratch Foundations**: Implementing Bag of Words and TF-IDF from scratch makes clear why text representations produce high-dimensional sparse matrices and how normalization prevents document length bias.
3. **Pipeline Integrity**: Using Scikit-Learn `Pipeline` guarantees that feature extraction parameters (vocabulary, IDF weights) remain strictly isolated to training folds.
4. **Beyond Accuracy**: In spam filtering, False Positives carry a vastly higher operational cost than False Negatives; tuning models for high precision is essential.
