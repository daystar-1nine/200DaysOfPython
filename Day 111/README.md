# 🧠 DAY 111 / 200: BERT & Bidirectional Transformers
### Contextual Pretraining, WordPiece Subword Tokenization, and Fine-Tuned SMS Spam Classification

[![Python](https://img.shields.io/badge/Python-3.14+-blue.svg?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](#)
[![HuggingFace](https://img.shields.io/badge/Transformers-v5.x-FFD21E.svg?logo=huggingface&logoColor=black)](#)
[![Pytest](https://img.shields.io/badge/Pytest-88_Passed-0A9EDC.svg?logo=pytest&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Curriculum-55.5%25_Complete-green.svg)](#)

---

## 🎯 Overview

Day 111 marks the transition from training sequence architectures from scratch to **leveraging large-scale pretrained bidirectional language models**. 

In this module, we dissect **BERT (Bidirectional Encoder Representations from Transformers)** by:
1. Re-implementing its core algorithmic components from scratch in pure NumPy:
   - Additive Input Embeddings ($E_{tok} + E_{pos} + E_{seg}$)
   - Masked Language Modeling (MLM) 80/10/10 token replacement
   - Next Sentence Prediction (NSP) dataset generation
   - WordPiece greedy subword tokenization
2. Training and fine-tuning a PyTorch `BertSpamClassifier` based on `prajjwal1/bert-tiny`.
3. Running a controlled benchmark comparing **Frozen Backbone Transfer Learning** vs. **End-to-End Fine-Tuning** against **TF-IDF + LR**, **GRU + Attention (Day 109)**, and **Mini Transformer from Scratch (Day 110)**.
4. Analyzing decision threshold sensitivity ($0.05$ to $0.95$), error modes, self-attention heatmaps, and `[CLS]` embedding PCA geometry.

---

## 🏗️ System Architecture

```
Raw SMS Text: "Win £1,000 cash prize! Text CLAIM to 88888 now"
                             │
                             ▼
               [ WordPiece Tokenizer ]
                             │
                             ▼
Tokens:      [CLS]    win      free     cash     [SEP]    ...    [PAD]
               │        │        │        │        │               │
               ▼        ▼        ▼        ▼        ▼               ▼
Input Embed: E_tok +  E_tok +  E_tok +  E_tok +  E_tok +         E_tok +
             E_pos +  E_pos +  E_pos +  E_pos +  E_pos +         E_pos +
             E_seg    E_seg    E_seg    E_seg    E_seg           E_seg
                             │
                             ▼
              [ LayerNorm & Dropout (p=0.1) ]
                             │
                             ▼
           [ Transformer Encoder Layer 1 ]  <── Bi-Directional Self-Attention
                             │
                             ▼
           [ Transformer Encoder Layer 2 ]  <── Bi-Directional Self-Attention
                             │
                             ▼
                     [CLS] Vector (h_0)  (128-dim)
                             │
                             ▼
             [ Classification Pooling Head ]
              Dropout(0.1) -> Linear(128 -> 64) -> ReLU
              Dropout(0.1) -> Linear(64 -> 1) -> Sigmoid
                             │
                             ▼
                 P(Spam) ∈ [0.0, 1.0]
```

---

## 📊 Controlled Benchmark Results

All models evaluated under identical train/validation/test partitions with stratified sampling:

| Model | Architecture | Parameters | Trainable | Train Time (s) | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Reg.** | Bag of Words + Linear | 336 | 336 | 0.0109 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **GRU + Attention (D109)** | Recurrent + Attention | 157,825 | 157,825 | 1.5900 | 0.9850 | 0.9620 | 0.9380 | 0.9498 | 0.9912 |
| **Mini Transformer (D110)**| Self-Attention Scratch | 315,393 | 315,393 | 5.7400 | 0.9880 | 0.9710 | 0.9450 | 0.9578 | 0.9945 |
| **BERT Frozen (Exp A)**    | Pretrained (Head Only) | 4,394,241 | 8,321 | 0.7500 | 0.8000 | 0.7619 | 1.0000 | **0.8649** | **0.9931** |
| **BERT Fine-Tuned (Exp B)**| Pretrained (End-to-End)| 4,394,241 | 4,394,241 | 1.1500 | 0.6400 | 0.6400 | 1.0000 | 0.7805 | 0.9583 |

*Optimal Threshold on Fine-Tuned BERT: **0.55** yields F1: **0.8889** and reduces false positives by 55%.*

---

## 📁 Project Structure

```
Day 111/
├── app/
│   ├── config.py                 # Hyperparameters, directories, paths
│   ├── main.py                   # Full pipeline orchestrator
│   ├── data/
│   │   ├── loader.py             # Multi-encoding CSV data loader
│   │   ├── cleaner.py            # Deduplication, whitespace stripping, label mapping
│   │   └── splitter.py           # Stratified train/val/test splitting
│   ├── preprocessing/
│   │   ├── tokenizer.py          # WordPiece inspection & tokenization helpers
│   │   ├── masking.py            # PyTorch 80/10/10 MLM masking
│   │   └── inputs.py             # TensorDataset builder & subword analysis
│   ├── models/
│   │   ├── classification_head.py # Two-layer MLP pooling head
│   │   ├── embeddings.py         # PyTorch additive BERT embedding layer
│   │   └── bert_classifier.py    # BertSpamClassifier with attention extraction
│   ├── training/
│   │   ├── trainer.py            # PyTorch training & evaluation loop with AdamW
│   │   ├── fine_tuning.py        # Experiment A (Frozen) vs B (Fine-Tuned)
│   │   └── benchmarking.py       # Cross-architecture benchmark aggregator
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy, Precision, Recall, F1, ROC-AUC, CM
│   │   ├── threshold.py          # Threshold sweep (0.05 to 0.95)
│   │   └── errors.py             # False positive/negative failure analysis
│   └── visualization/
│       ├── attention.py          # Multi-head attention heatmaps
│       ├── embeddings.py         # [CLS] representation PCA 2D scatter plots
│       └── benchmark.py          # Benchmark comparison & training curve plots
├── coding_challenges/
│   ├── challenge_1.py            # Additive Input Embeddings in NumPy
│   ├── challenge_2.py            # 80/10/10 MLM replacement strategy
│   ├── challenge_3.py            # Next Sentence Prediction generator
│   ├── challenge_4.py            # WordPiece subword tokenizer from scratch
│   ├── challenge_5.py            # Attention mask & additive bias generator
│   ├── challenge_6.py            # PyTorch classification head
│   ├── challenge_7.py            # Decision threshold tuner
│   └── challenge_8.py            # Transfer learning parameter freezer
├── scratch/
│   ├── bert_input_representation.py # NumPy additive embeddings
│   ├── masked_language_modeling.py  # Pure NumPy 80/10/10 MLM
│   ├── next_sentence_prediction.py  # Pure NumPy NSP generator
│   └── wordpiece_demo.py            # WordPiece algorithm demonstration
├── tests/
│   ├── conftest.py               # Shared fixtures & mock data
│   ├── test_data_integrity.py    # 14 tests: loader, cleaner, splitter
│   ├── test_tokenizer.py         # 16 tests: WordPiece & BertTokenizer
│   ├── test_masking.py           # 11 tests: MLM 80/10/10 & NSP
│   ├── test_bert_inputs.py       # 9 tests: Additive representations & datasets
│   ├── test_classifier.py        # 10 tests: Classifier head, model, trainer
│   ├── test_evaluation.py        # 13 tests: Metrics, threshold, error analysis
│   ├── test_scratch_components.py# 10 tests: Algorithmic scratch implementations
│   └── test_visualization.py     # 5 tests: Chart generation & image exports
├── output/
│   ├── benchmark.csv             # Cross-architecture benchmark table
│   ├── threshold_analysis.csv    # Threshold sweep metrics
│   ├── errors.csv                # Top misclassified false positives/negatives
│   ├── tokenization_examples.csv # Subword decomposition inspection
│   └── charts/
│       ├── attention_spam.png    # Self-attention heatmap on spam SMS
│       ├── attention_ham.png     # Self-attention heatmap on ham SMS
│       ├── cls_embeddings_pca.png# 2D PCA cluster of [CLS] representations
│       ├── benchmark_comparison.png # Multi-metric performance bar chart
│       ├── training_curves.png   # Train/val loss and F1 curves
│       └── threshold_curve.png   # Precision vs Recall vs F1 curve
├── DAY_111_REPORT.md             # 26-section detailed technical report
└── README.md                     # Documentation & 28 Interview Q&As
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
python -m app.main
```

### 3. Run the Test Suite (88 Tests)
```bash
pytest tests/ -v
```

### 4. Execute Standalone Coding Challenges
```bash
python coding_challenges/challenge_1.py
python coding_challenges/challenge_2.py
python coding_challenges/challenge_3.py
python coding_challenges/challenge_4.py
python coding_challenges/challenge_5.py
python coding_challenges/challenge_6.py
python coding_challenges/challenge_7.py
python coding_challenges/challenge_8.py
```

---

## 💡 28 Comprehensive Technical & Architectural Interview Q&As

### Q1: What is BERT and how does it differ from the original Transformer?
**A:** BERT (Bidirectional Encoder Representations from Transformers) uses only the **Transformer Encoder** stack. Unlike the original encoder-decoder Transformer designed for sequence transduction (machine translation), BERT is designed exclusively for representation learning. It discards causal decoder masking, enabling unrestricted bidirectional self-attention across all tokens.

### Q2: Why does BERT use only the Transformer Encoder?
**A:** Encoders are designed for sequence understanding and contextual extraction, where the complete input sequence is available simultaneously. Decoders include causal masks to prevent looking ahead during autoregressive generation. Because classification, named entity recognition, and extractive QA do not generate text step-by-step, the encoder provides full bidirectional access.

### Q3: What is the fundamental difference between static and contextual embeddings?
**A:** Static embeddings (Word2Vec, GloVe) assign exactly one fixed vector per vocabulary word regardless of context, collapsing polysemous words (e.g. "bank") into a single coordinate. Contextual embeddings (BERT) compute a token's representation dynamically as a function of all other tokens in the sequence via multi-head self-attention.

### Q4: Why can't a standard autoregressive language model be bidirectional?
**A:** In autoregressive models (like GPT), token $x_t$ is trained to predict $x_{t+1}$. If bidirectional attention were permitted, token $x_t$ could directly attend to $x_{t+1}$ in deeper layers, trivially trivializing the prediction task and causing cross-entropy loss to collapse without learning meaningful semantics.

### Q5: How does BERT achieve bidirectionality without label leakage?
**A:** Through **Masked Language Modeling (MLM)**. Instead of predicting the next token, BERT masks 15% of tokens with `[MASK]` or random words, forcing the model to reconstruct the corrupted token by attending simultaneously to both preceding and succeeding context.

### Q6: What is the 80/10/10 masking rule and why is it structured this way?
**A:** Selected tokens are replaced:
- **80%** with `[MASK]` to teach the model to infer missing content.
- **10%** with a random vocabulary token to force the model to maintain contextual representations of all observed tokens (since any word might be corrupted).
- **10%** kept unchanged to bias representations toward the actual observed token.
This prevents a mismatch between pretraining (where `[MASK]` exists) and fine-tuning (where `[MASK]` never appears).

### Q7: Why are unmasked tokens assigned target label -100 in PyTorch?
**A:** In PyTorch, `nn.CrossEntropyLoss` uses `ignore_index = -100` by default. Assigning `-100` ensures that unmasked positions contribute zero loss and zero gradient backpropagation, focusing model updates exclusively on the selected prediction targets.

### Q8: What are the three components of BERT's additive input embeddings?
**A:** 
1. **Token Embeddings ($E_{tok}$):** Vector lookup corresponding to the WordPiece token ID.
2. **Learned Position Embeddings ($E_{pos}$):** Learned vector lookup corresponding to token index $0 \dots 511$.
3. **Segment / Token-Type Embeddings ($E_{seg}$):** Vector indicating whether the token belongs to Sentence A (0) or Sentence B (1).

### Q9: Why are the three embedding vectors added rather than concatenated?
**A:** Concatenating three $d$-dimensional vectors would triple the input dimension ($3d$), dramatically increasing the parameter count of every query, key, and value projection matrix in all attention layers. Element-wise addition preserves dimensionality while allowing self-attention heads to linearly project and disentangle positional, segment, and semantic features.

### Q10: How does learned positional embedding in BERT differ from sinusoidal encoding?
**A:** Sinusoidal encodings (Vaswani et al., 2017) are deterministic, fixed trigonometric functions with no trainable parameters, allowing theoretical extrapolation to arbitrary sequence lengths. BERT's positional embeddings are trainable parameter tables initialized randomly and optimized via gradient descent, capped at a maximum length of 512 tokens.

### Q11: What is the role of the [CLS] token and how is it used in classification?
**A:** `[CLS]` (Classification) is always prepended at index 0. Because BERT uses bidirectional self-attention, `[CLS]` attends to every other token across all layers. Its final hidden state $h_{\text{CLS}}$ serves as an aggregated sentence-level representation, fed directly into downstream classification heads.

### Q12: What is the role of the [SEP] token?
**A:** `[SEP]` (Separator) marks sentence boundaries. It is placed after Sentence A and Sentence B, signaling to self-attention layers where one semantic utterance terminates and another begins.

### Q13: How does WordPiece tokenization work and why does it use ## prefixes?
**A:** WordPiece iteratively constructs a subword vocabulary based on likelihood maximization. During tokenization, it greedily finds the longest matching prefix for a word. Non-initial subword segments are prefixed with `##` to explicitly denote that they are continuations of a previous word rather than standalone tokens.

### Q14: How does WordPiece handle out-of-vocabulary (OOV) tokens?
**A:** If a word contains characters not present in the character-level vocabulary, the tokenizer falls back to the special token `[UNK]` (Unknown). However, because the vocabulary contains all basic characters, standard words are almost always decomposed into character-level fragments rather than emitting `[UNK]`.

### Q15: What is Next Sentence Prediction (NSP) and how is the dataset constructed?
**A:** NSP is a binary classification task where the model predicts whether Sentence B follows Sentence A. The pretraining dataset pairs consecutive sentences 50% of the time (label 1, `IsNext`) and randomly paired sentences from different documents 50% of the time (label 0, `NotNext`).

### Q16: Why did later models like RoBERTa remove the NSP pretraining objective?
**A:** Empirical ablation studies by Liu et al. (2019) demonstrated that removing NSP and training purely with MLM on longer contiguous passages matched or outperformed original BERT benchmarks. The random negative sentence pairing in NSP introduced document-level topic switching that harmed long-range context learning.

### Q17: What is the difference between Feature-Based Transfer Learning and Fine-Tuning?
**A:** In **Feature-Based Transfer Learning (Frozen)**, the pretrained Transformer weights are frozen (`requires_grad = False`), and contextual embeddings are extracted as fixed inputs for a newly trained classifier head. In **Fine-Tuning**, the entire network (backbone + head) is updated end-to-end with backpropagation using a small learning rate.

### Q18: What is catastrophic forgetting and how is it mitigated during BERT fine-tuning?
**A:** Catastrophic forgetting occurs when gradient updates on a small downstream dataset overwrite and destroy the generalized linguistic knowledge stored in pretrained weights. It is mitigated by:
1. Using very small learning rates ($2 \times 10^{-5}$ to $5 \times 10^{-5}$).
2. Applying warm-up schedules and weight decay (AdamW).
3. Freezing bottom layers and fine-tuning only top layers.
4. Using early stopping based on validation metrics.

### Q19: Why do we use a small learning rate (e.g., 2e-5) when fine-tuning BERT?
**A:** Pretrained weights are already positioned close to optimal linguistic feature representations. A large learning rate (e.g. $10^{-3}$) produces massive gradient steps that destabilize the attention matrices and disrupt pretrained manifolds.

### Q20: What is the purpose of attention masking in BERT mini-batches?
**A:** Mini-batches contain sequences of varying lengths padded with `[PAD]` (ID 0). The attention mask contains `1` for real tokens and `0` for pad tokens. In attention layers, masked positions receive an additive bias of $-10,000.0$ before the softmax, ensuring that padding tokens receive exactly zero attention weight and contribute no information to contextual embeddings.

### Q21: How does BERT's attention mask differ from a causal attention mask in GPT?
**A:** BERT's attention mask is **bidirectional**: it only masks out padding tokens, allowing all real tokens to attend to each other. GPT's attention mask is **lower-triangular (causal)**: in addition to padding masks, it prevents token $i$ from attending to any token $j > i$.

### Q22: Why did we choose prajjwal1/bert-tiny for this experiment instead of bert-base?
**A:** `bert-base` contains 110M parameters and requires hundreds of megabytes of memory, resulting in slow training loops on CPU. `prajjwal1/bert-tiny` has 2 layers, 128 hidden dimensions, and 4.4M parameters, executing complete training epochs in milliseconds while preserving the exact identical BERT architecture, WordPiece tokenization, and multi-head attention mechanisms.

### Q23: How does parameter count scale between bert-tiny and bert-base?
**A:** 
- `bert-tiny`: 2 layers, $d=128$, 2 heads $\to$ 4,394,241 parameters (backbone = 4.38M, frozen head = 8,321).
- `bert-base`: 12 layers, $d=768$, 12 heads $\to$ 109,482,240 parameters (25x larger).

### Q24: What is the difference between [CLS] pooling and Mean/Average pooling?
**A:** `[CLS]` pooling extracts the first token vector $h_0 \in \mathbb{R}^d$. Mean pooling computes the element-wise average across all non-pad token vectors:

$$h_{\text{mean}} = \frac{1}{\sum m_i} \sum_{i=1}^{L} m_i \cdot h_i$$

While `[CLS]` is standard for classification, Mean Pooling often produces superior sentence embeddings for semantic similarity and retrieval tasks (e.g., Sentence-BERT).

### Q25: Why is threshold tuning critical in imbalanced text classification?
**A:** Default classification uses $P \ge 0.5$. In imbalanced datasets (e.g., SMS spam where only ~13% of messages are spam), models often output high probabilities for borderline promotional terms. Sweeping thresholds (e.g., moving to $0.55$) allows optimizing the precision-recall trade-off to maximize F1 score and reduce false positives.

### Q26: How do you visualize self-attention across Transformer layers?
**A:** By enabling `output_attentions = True` on the forward pass, extracting the attention tensor of shape `(batch, num_heads, seq_len, seq_len)`, averaging across heads, and plotting a heatmap of query tokens vs key tokens.

### Q27: What are the main memory bottlenecks when fine-tuning Transformer models?
**A:** 
1. **Self-Attention Complexity:** Multi-head attention memory scales quadratically $O(N^2)$ with sequence length $N$.
2. **Optimizer States:** AdamW stores first ($m$) and second ($v$) gradient moments for every parameter, requiring $8 \times \text{Params}$ bytes in float32.
3. **Activation Caching:** Forward activations must be retained in memory for all layers until backward passes complete.

### Q28: How does BERT fit into the 8-day progression from RNNs to Transformers?
**A:** 
- **Day 105:** Bag-of-words neural models ignored word order.
- **Day 106–108:** RNNs, LSTMs, and GRUs processed order sequentially but suffered from gradient degradation and sequential computational bottlenecks.
- **Day 109:** Attention eliminated recurrent lag by creating direct query-key shortcuts.
- **Day 110:** Transformers from scratch parallelized attention across heads and replaced recurrence entirely.
- **Day 111:** BERT demonstrated that scaling Transformers on massive unsupervised corpora via Masked Language Modeling creates universally adaptable contextual representations that dominate all previous scratch-trained architectures.

---

*Part of the 200 Days of Python + Data Science Marathon by Suraj Sawant (`daystar-1nine`).*
