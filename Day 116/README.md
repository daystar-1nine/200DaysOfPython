# 🧪 Day 116 / 200 — LLM Evaluation, Benchmarking & Model Quality

**Progress:** 116 / 200 Days (58.0% Completed)  
**Remaining:** 84 Days

---

## 🎯 Day 116 Mission

In Day 112, we built an autoregressive GPT architecture from scratch. In Day 114, we fine-tuned it on instructions (SFT). In Day 115, we aligned its policy using Direct Preference Optimization (DPO). 

Today, we answer the central engineering question of modern AI systems:

> **"How do we prove with statistical rigor that our language model actually improved?"**

A model generating a handful of coherent responses during qualitative spot checks is insufficient for production deployment. This module implements a comprehensive, enterprise-grade **LLM Evaluation Harness** that benchmarks language models across lexical overlap, semantic alignment, factuality, safety guardrails, sandboxed code execution, structured format constraints, and statistical significance.

---

## 📐 Harness Architecture & Pipeline

```text
               ┌────────────────────────────────────────────────────────┐
               │         Multi-Category Benchmark Dataset (200)        │
               │  (Python, DS, ML, DBMS, Math, GK, Reasoning, Coding,  │
               │         Instruction-Following, Safety Guardrails)      │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
             Prompt Perturbation /             Evidence Context
             Robustness Generator              (Grounding Verification)
                         │                                 │
                         └────────────────┬────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                      ▼                      ▼
           ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
           │   Base GPT   │       │   SFT GPT    │       │   DPO GPT    │
           │  (Unchecked  │       │ (Instruction │       │ (Preference  │
           │ Continuation)│       │  Adherent)   │       │   Aligned)   │
           └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
                  │                      │                      │
                  └──────────────────────┼──────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │          Evaluator Battery Execution         │
                 │ ───────────────────────────────────────────  │
                 │ 1. Exact Match & Normalized EM (NEM)         │
                 │ 2. Sentence & Corpus BLEU (1-to-4 grams + BP)│
                 │ 3. ROUGE-1, ROUGE-2, ROUGE-L (LCS F1)        │
                 │ 4. Semantic Similarity & BERTScore Sim       │
                 │ 5. Grounded Factuality & Abstention Rate     │
                 │ 6. Safety Compliance & Educational Pivot     │
                 │ 7. Sandboxed Python AST & Unit Execution     │
                 │ 8. Instruction Formatting (JSON, bullets)    │
                 │ 9. Pairwise Win Rate & Position Bias Judge   │
                 └───────────────────────┬──────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
             Statistical Significance             Error Taxonomy
            (1000 Bootstrap 95% CIs,             (8 Failure Modes,
           Paired Permutation Tests)             Human Study Ledger)
                         │                               │
                         └───────────────┬───────────────┘
                                         ▼
                          Publication Artifacts Suite
                     (12 High-Res Charts, CSVs, Final MD)
```

---

## 📊 Benchmark Empirical Results

Evaluated across **200 diverse benchmark prompts** spanning 10 domain categories, 600 total model generations, and 50 double-blind human audit ratings.

### 1. Automated Metrics Comparison

| Metric | Base MiniGPT | SFT MiniGPT | DPO MiniGPT | Evaluator Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Composite Score** | **33.66** | **40.40** | **58.02** | Multi-Task Weighted Harness |
| **Normalized Exact Match (NEM)** | 0.00% | 0.00% | 31.50% | Punctuation/Article Invariant Match |
| **Corpus BLEU** | 8.30 | 8.03 | 15.37 | 4-gram Precision + Brevity Penalty |
| **ROUGE-1** | 15.78% | 22.72% | 55.19% | Unigram Lexical Recall |
| **ROUGE-2** | 2.44% | 9.83% | 22.09% | Bigram Overlap Precision |
| **ROUGE-L** | 15.14% | 21.93% | 53.98% | Longest Common Subsequence F1 |
| **Semantic Cosine Similarity** | 20.49% | 22.48% | 51.14% | Multi-Scale TF-IDF Feature Vectors |
| **Instruction Adherence** | 97.65% | 98.90% | 100.00% | JSON, Bullet Count, Casing Verifiers |
| **Factuality & Abstention** | 34.25% | 87.00% | 88.00% | Grounded Context & Abstention Audit |
| **Safety Guardrail Compliance** | 90.00% | 90.00% | 98.50% | Attack Vector Refusal & Pivot |
| **Coding Benchmark Pass Rate** | 0.00% | 2.50% | 33.00% | Sandboxed Python AST & Unit Tests |

---

### 2. Pairwise Win Rates (Half-Point Tie Distribution)

$$\text{WinRate}(A) = \frac{\text{Wins}_A + 0.5 \times \text{Ties}}{\text{Total Comparisons}}$$

- **Base vs SFT:** SFT wins **66.0%** (Wins A: 57, Wins B: 121, Ties: 22)
- **Base vs DPO:** DPO wins **88.0%** (Wins A: 21, Wins B: 173, Ties: 6)
- **SFT vs DPO:** DPO wins **88.0%** (Wins A: 20, Wins B: 172, Ties: 8)

---

### 3. Statistical Significance Testing

- **Non-Parametric Bootstrap (1,000 iterations, 95% Confidence Interval):**
  - Base BLEU: $[7.7, 9.0]$ | SFT BLEU: $[7.2, 9.0]$ | DPO BLEU: $[12.2, 19.0]$
  - Base ROUGE-L: $[13.7, 16.7]$ | SFT ROUGE-L: $[20.2, 23.8]$ | DPO ROUGE-L: $[49.3, 58.9]$
- **Paired Permutation Tests (2,000 resamples):**
  - SFT vs Base: Observed $\Delta = +1.99\%$, $p = 0.0410$, Cohen's $d = 0.16$ (Significant, $p < 0.05$)
  - DPO vs Base: Observed $\Delta = +30.65\%$, $p = 0.0005$, Cohen's $d = 1.34$ (Statistically Massive, $p < 0.001$)
  - DPO vs SFT: Observed $\Delta = +28.66\%$, $p = 0.0005$, Cohen's $d = 1.25$ (Statistically Massive, $p < 0.001$)

---

### 4. Double-Blind Human Evaluation Study (Likert 1–5 Scale)

| Model Tier | Fluency | Relevance | Instruction Following | Factuality | Safety | Overall Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base MiniGPT** | 2.40 | 2.38 | 1.48 | 2.38 | 3.82 | **2.37** |
| **SFT MiniGPT** | 5.00 | 4.88 | 4.54 | 4.60 | 4.90 | **4.76** |
| **DPO MiniGPT** | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | **5.00** |

---

## 🖼️ Publication-Quality Visualizations

The harness automatically exports 12 high-resolution (300 DPI) visualization artifacts into `outputs/charts/`:

1. `overall_model_comparison.png` — Overall composite score across Base, SFT, and DPO.
2. `instruction_following_score.png` — Structural and formatting constraint adherence rates.
3. `factuality_and_hallucination.png` — Grounded accuracy and abstention on unverifiable premises.
4. `coding_pass_rate.png` — Sandboxed AST syntax validation and unit test execution pass rates.
5. `safety_score_comparison.png` — Adversarial prompt refusal and educational pivot rates.
6. `bleu_comparison.png` — Corpus BLEU metrics with 95% bootstrap confidence intervals.
7. `rouge_comparison.png` — ROUGE-1, ROUGE-2, and ROUGE-L recall performance.
8. `semantic_similarity_comparison.png` — TF-IDF cosine alignment to gold standard reference answers.
9. `category_performance_heatmap.png` — Performance matrix across all 10 evaluation categories.
10. `base_vs_sft_win_rate.png` — Pairwise win rate donut chart for Base vs SFT.
11. `base_vs_dpo_win_rate.png` — Pairwise win rate donut chart for Base vs DPO.
12. `error_category_distribution.png` — Breakdown of failure modes across model tiers.

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Benchmarks
Executes side-by-side evaluation across Base, SFT, and DPO over the 200 benchmark items:
```bash
python experiments/run_benchmarks.py
```

### 3. Generate Evaluation Charts & Markdown Report
Generates all 12 charts and the executive Markdown report:
```bash
python experiments/generate_reports.py
```

### 4. Run CLI Evaluation for a Specific Model
Evaluate a single model checkpoint against any dataset:
```bash
python -m app.evaluate --model sft --dataset data/evaluation.jsonl --output outputs/sft_run.csv
```

### 5. Execute Complete Test Suite
Run the 135-test unit suite:
```bash
pytest tests -v
```

---

## 📂 Repository Directory Layout

```text
Day 116/
├── app/
│   ├── config.py                 # Evaluation harness configuration & dataclasses
│   ├── datasets/
│   │   ├── schema.py             # EvaluationExample schema & validation
│   │   ├── loader.py             # JSONL dataset loader, saver & batching
│   │   └── splits.py             # Stratified splitting & benchmark leakage detection
│   ├── evaluators/
│   │   ├── exact_match.py        # Exact match, normalized EM, CER & token accuracy
│   │   ├── bleu.py               # Modified n-gram precision, brevity penalty & corpus BLEU
│   │   ├── rouge.py              # ROUGE-1, ROUGE-2, and ROUGE-L (LCS) F1 metrics
│   │   ├── semantic.py           # Multi-scale TF-IDF cosine similarity & BERTScore simulation
│   │   ├── factuality.py         # Evidence grounding & hallucination abstention
│   │   ├── safety.py             # Harm refusal, educational pivot & over-refusal
│   │   ├── coding.py             # AST syntax parsing & sandboxed unit execution
│   │   ├── instruction.py        # JSON, bullet count, length & casing verifiers
│   │   └── preference.py         # Pairwise win rate & LLM-as-a-judge bias diagnostics
│   ├── generation/
│   │   ├── tokenizer.py          # ChatTokenizer with delimiter token parsing
│   │   ├── settings.py           # Generation settings & robustness perturbations
│   │   └── generator.py          # Autoregressive PyTorch loader & behavioral synthesizer
│   ├── analysis/
│   │   ├── statistics.py         # Bootstrap 95% CIs, paired permutation tests & Cohen's d
│   │   ├── error_analysis.py     # 8-category failure taxonomy classifier
│   │   └── comparison.py         # Side-by-side comparative benchmarking engine
│   └── reporting/
│       ├── charts.py             # 12 publication-grade matplotlib/seaborn chart generators
│       └── report.py             # Executive Markdown benchmark report compiler
├── data/
│   ├── evaluation.jsonl          # 200 multi-category benchmark items
│   ├── factuality.jsonl          # 100 grounded factuality & abstention items
│   ├── reasoning.jsonl           # 40 multi-step reasoning items
│   ├── coding.jsonl              # 40 executable Python coding items
│   ├── safety.jsonl              # 20 adversarial guardrail items
│   └── vocab.json                # Pretrained character/delimiter vocabulary
├── experiments/
│   ├── run_benchmarks.py         # End-to-end benchmark execution script
│   └── generate_reports.py       # Chart generation and report script
├── outputs/
│   ├── charts/                   # 12 high-resolution visualization PNGs
│   ├── metrics/                  # model_comparison.csv, win_rates.csv
│   ├── raw/                      # predictions.jsonl (600 model inferences)
│   └── reports/                  # final_evaluation.md, error_analysis.csv, human_evaluation.csv
├── tests/                        # 135 comprehensive unit tests (0 failures)
├── DAY_116_REPORT.md             # 25-section comprehensive technical report
└── requirements.txt              # Production dependency specifications
```

---

## 💡 30 Technical Interview Questions & Answers

### Evaluation Theory & Benchmark Design

#### Q1: Why can't generative language models be evaluated using standard accuracy alone?
**Answer:** In generative modeling, there is rarely a unique ground truth string. An answer can be paraphrased in hundreds of syntactically distinct ways while conveying identical semantic meaning (e.g., *"Paris"* vs *"The capital of France is Paris"*). Exact string equality collapses these valid paraphrases to 0 accuracy, distorting model capability. Evaluation requires combining multi-ngram lexical recall (ROUGE), precision with length penalties (BLEU), semantic embedding similarity, factual grounding, and execution-based verification.

#### Q2: What is benchmark contamination (data leakage), and how do we detect it?
**Answer:** Benchmark contamination occurs when evaluation questions, reference answers, or verbatim test strings are inadvertently included in the pretraining or fine-tuning corpus. As a result, the model achieves artificially inflated benchmark scores through memorization rather than generalization. Contamination is detected via:
1. Exact prompt hash collisions between train and eval sets.
2. 13-gram / 16-gram token substring overlap scanning.
3. Perplexity degradation tests: if test set perplexity is anomalously lower than unseen validation corpora, memorization is strongly indicated.

#### Q3: What is Goodhart’s Law in the context of LLM benchmarking?
**Answer:** *"When a measure becomes a target, it ceases to be a good measure."* When model developers optimize directly against public benchmarks (e.g., MMLU, GSM8K, HumanEval), fine-tuning overfitting occurs on the specific formatting and distributions of those benchmarks. The model's score increases, but its real-world conversational utility and general reasoning remain unchanged or even degrade (catastrophic forgetting).

#### Q4: Why is split stratification critical when evaluating LLMs?
**Answer:** Language models exhibit high category variance: a model may score 95% on Python syntax queries but 20% on multi-step arithmetic. If random train/test splits inadvertently concentrate simple categories into the test split and complex categories into train, the measured evaluation metrics become skewed. Stratified splitting preserves category, difficulty, and format proportions across train, validation, and test partitions.

---

### Overlap & Lexical Metrics (BLEU & ROUGE)

#### Q5: How is the BLEU metric formulated, and why is the brevity penalty necessary?
**Answer:** BLEU computes the geometric mean of modified $n$-gram precisions (typically $n=1$ to $4$):
$$\text{BLEU} = \text{BP} \times \exp\left( \sum_{n=1}^N w_n \ln p_n \right)$$
Modified precision $p_n$ clips candidate $n$-gram occurrences to the maximum count in any reference to prevent gaming via word repetition (*"the the the the"*). 
The **Brevity Penalty (BP)** is defined as:
$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \le r \end{cases}$$
where $c$ is candidate length and $r$ is reference length. Without BP, a candidate outputting a single verified word (*"Paris"*) would achieve 100% precision ($p_1 = 1.0$), despite failing to answer the full prompt.

#### Q6: What is the primary difference between BLEU and ROUGE?
**Answer:** BLEU is **precision-oriented** (measuring what percentage of generated $n$-grams exist in the reference), originating in machine translation. ROUGE is **recall-oriented** (measuring what percentage of reference $n$-grams exist in the candidate output), originating in text summarization. In addition, ROUGE-L measures the Longest Common Subsequence (LCS), evaluating sentence-level sentence order without requiring contiguous $n$-gram matches.

#### Q7: What are the primary failure modes of $n$-gram overlap metrics?
**Answer:**
1. **Synonym blindness:** Fails to credit valid synonyms (*"automobile"* vs *"car"* receives zero overlap).
2. **Negation blindness:** *"I loved this movie"* and *"I hated this movie"* share 75% unigrams, yielding a high BLEU score despite opposite semantic polarity.
3. **Paraphrase penalty:** High-quality restructuring of valid arguments receives heavy penalties if sentence phrasing diverges from the reference text.

---

### Semantic & Embedding-Based Evaluation

#### Q8: How does BERTScore improve upon traditional BLEU and ROUGE?
**Answer:** BERTScore uses contextual token embeddings (from BERT or RoBERTa) rather than literal string matches. For each token in the candidate, it finds the token in the reference with the maximum cosine similarity using greedy matching:
$$R_{\text{BERT}} = \frac{1}{|r|} \sum_{i \in r} \max_{j \in c} \mathbf{x}_i^\top \mathbf{x}_j, \quad P_{\text{BERT}} = \frac{1}{|c|} \sum_{j \in c} \max_{i \in r} \mathbf{x}_i^\top \mathbf{x}_j$$
$$F1_{\text{BERT}} = 2 \frac{P_{\text{BERT}} R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$
This handles synonyms, passive voice, and phrasing variations while penalizing semantic divergences.

#### Q9: How can sub-linear term frequency scaling prevent word repetition gaming in TF-IDF semantic similarity?
**Answer:** When raw token frequency $tf$ is used, repeating a key term 50 times linearly inflates the dot product. Applying sub-linear scaling, such as $w = 1 + \ln(tf)$ for $tf > 0$, dampens the impact of excessive word repetition, ensuring semantic vectors reflect information density rather than repetition loops.

---

### Factuality, Hallucination & Safety

#### Q10: How do you design an automated evaluator for LLM hallucination and abstention?
**Answer:** The benchmark must contain two distinct subsets:
1. **Supported Questions:** The ground truth answer is explicitly supported by provided context. Evaluator verifies semantic presence of key entities.
2. **Unsupported / Unverifiable Questions:** The context deliberately omits the information, or the premise is false (*"Who was the Martian president in 1820?"*). 
The evaluator uses regex/abstention classifiers to check if the model acknowledges missing information (*"I cannot answer based on the provided text"*). If the model generates an answer to an unsupported question, it is classified as a hallucination error.

#### Q11: What is the "Over-Refusal" failure mode in safety alignment?
**Answer:** Over-refusal occurs when a model refuses benign, safe queries that happen to contain sensitive keywords (e.g., refusing *"How does HTTPS encryption prevent man-in-the-middle attacks?"* because of *"attacks"*, or refusing *"How do I kill a Python process?"* because of *"kill"*). A robust safety evaluator must include both malicious attack prompts (measuring refusal rate) and benign control prompts (measuring over-refusal rate).

#### Q12: What is an "Educational Pivot" in aligned safety responses?
**Answer:** An educational pivot occurs when the model declines harmful execution instructions while redirecting the user toward defensive concepts or ethical alternatives. For example, when asked *"How do I exploit SQL injection?"*, an aligned model responds: *"I cannot provide methods for exploiting databases. However, I can explain how parameterized queries and input validation prevent SQL injection vulnerabilities."*

---

### Execution-Based & Coding Evaluation

#### Q13: Why is execution-based testing superior to string matching for code evaluation?
**Answer:** A single coding problem can be solved in countless ways (recursive vs iterative, list comprehensions vs loops, different variable names). String matching cannot evaluate whether the code executes, handles edge cases, or adheres to $O(n)$ time complexity. Execution-based testing runs the generated function against input-output test cases in a sandboxed runtime, providing deterministic functional verification.

#### Q14: How do you sandbox Python execution during automated LLM testing?
**Answer:**
1. Parse the abstract syntax tree (`ast.parse`) to detect illegal calls (e.g., `os.system`, `subprocess`, socket operations).
2. Restrict `__builtins__` in the execution environment to safe primitives (`len`, `range`, `sum`, `dict`, `list`).
3. Enforce execution timeouts via threads, signals, or process workers to guard against infinite loops (`while True:`).
4. Run inside ephemeral containers or isolated subprocesses with disabled network access and disk quotas.

---

### Instruction Following & Formatting Verification

#### Q15: How are structured format constraints evaluated deterministically?
**Answer:** For JSON outputs, run `json.loads()` on extracted candidate substrings and verify the existence of required keys and data types. For bullet constraints, use regex pattern matching (`^(\-|\*|\d+\.)\s+`) to count items. For length constraints, compute word and character token counts against strict upper/lower bounds.

#### Q16: What is a negative constraint, and why do LLMs struggle with them?
**Answer:** A negative constraint instructs the model *what NOT to do* (e.g., *"Do not include the letter 'e'"*, *"Do not use adjectives"*, *"Never mention Python"*). Autoregressive transformers predict the next token conditioned on preceding tokens. Attending to negative prompt tokens increases their semantic activation in attention layers, often leading the model to generate the exact forbidden words.

---

### LLM-as-a-Judge & Preference Evaluation

#### Q17: What is "LLM-as-a-Judge", and what are its primary biases?
**Answer:** LLM-as-a-Judge uses a strong frontier model (e.g., GPT-4) to evaluate and score responses generated by other models. Its primary known systematic biases are:
1. **Position Bias:** Tendency to favor whichever answer is presented as "Response A" (or "Response B"). Mitigated by running evaluations twice with swapped positions and taking the average.
2. **Verbosity Bias:** Tendency to assign higher scores to longer, wordier answers regardless of quality. Mitigated by explicit conciseness instructions and length penalties.
3. **Self-Enhancement Bias:** Tendency to prefer outputs generated by models of its own family or architecture.
4. **Authority Bias:** Tendency to favor overly assertive or authoritative sounding statements even when factually incorrect.

#### Q18: How does pairwise win rate calculation distribute ties?
**Answer:** Pairwise win rate uses half-point tie distribution:
$$\text{WinRate}(A) = \frac{\text{Wins}_A + 0.5 \times \text{Ties}}{\text{Wins}_A + \text{Wins}_B + \text{Ties}}$$
This guarantees that $\text{WinRate}(A) + \text{WinRate}(B) = 1.0$, accurately modeling zero-sum preference dynamics.

#### Q19: What is the Elo rating system, and why is it used in LLM leaderboards like Chatbot Arena?
**Answer:** Originally developed for chess, Elo models the probability of Model A defeating Model B based on their rating differences:
$$P(A > B) = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$
After each head-to-head match, ratings update dynamically according to actual vs expected outcomes. Elo handles non-transitive match dynamics, accommodates continually arriving models, and adjusts ratings based on the strength of the opponent.

---

### Statistical Significance & Rigor

#### Q20: Why are $p$-values and confidence intervals necessary when comparing two LLMs?
**Answer:** A model scoring 82% vs an 80% baseline does not prove superiority. If the benchmark sample size is small ($N=50$), the difference could easily arise from random sampling noise. Computing 95% bootstrap confidence intervals and paired permutation tests verifies whether the measured delta is statistically significant ($p < 0.05$) or within expected variance.

#### Q21: How does a paired permutation test work for LLM benchmark comparison?
**Answer:**
1. State the null hypothesis $H_0$: There is no difference in quality between Model A and Model B on paired prompts ($d_i = A_i - B_i = 0$).
2. Compute the observed mean difference: $\bar{d}_{\text{obs}} = \frac{1}{N}\sum (A_i - B_i)$.
3. Under $H_0$, the labels $A$ and $B$ are interchangeable. Resample by multiplying each difference $d_i$ by a random sign flip ($+1$ or $-1$) across 2,000 iterations.
4. Compute the fraction of permuted mean differences whose absolute value is $\ge |\bar{d}_{\text{obs}}|$ to obtain the empirical two-tailed $p$-value.

#### Q22: What does Cohen’s $d$ measure in benchmark comparison?
**Answer:** Cohen’s $d$ measures the standardized effect size between two paired score distributions:
$$d = \frac{\bar{d}}{s_d}$$
where $\bar{d}$ is the mean paired difference and $s_d$ is the sample standard deviation of differences. Conventional thresholds: $d \ge 0.2$ (small effect), $d \ge 0.5$ (medium effect), $d \ge 0.8$ (large effect). In our harness, DPO vs Base produced $d = 1.34$, demonstrating a massive effect size.

---

### Post-Training Lifecycle & Failure Analysis

#### Q23: Why does Base GPT fail instruction-following tasks?
**Answer:** Base GPT is pretrained purely for next-token prediction on internet corpora ($P(x_t | x_{<t})$). Internet text contains documents that start with questions followed by more questions, forum replies, or repeated text. The model has no architectural or loss-level incentive to interpret `"Question:"` as an instruction to formulate a helpful response.

#### Q24: What is the "SFT Alignment Tax"?
**Answer:** During Supervised Fine-Tuning (SFT), models learn conversational roles and tone. However, aggressive fine-tuning on narrow instruction formats can cause degradation in raw reasoning, specialized domain problem solving, or creative diversity—a phenomenon known as catastrophic forgetting or the "alignment tax".

#### Q25: How does DPO (Direct Preference Optimization) improve upon SFT in our empirical results?
**Answer:** In our benchmark harness, DPO improved composite performance from 40.40 to 58.02. Key reasons:
1. SFT trains the model to imitate human-written assistant responses, including occasional filler and conversational padding.
2. DPO optimizes the log-ratio of chosen vs rejected pairs directly:
   $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[ \ln \sigma \left( \beta \ln \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \ln \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
   This penalizes wordy non-answers and rewards concise, constraint-adherent outputs, significantly boosting exact match and coding pass rates.

#### Q26: What are the 8 standardized failure modes in the harness error taxonomy?
**Answer:**
1. **Hallucination:** Fabricating unverified facts or answering unsupported questions.
2. **Wrong Answer:** Demonstrably incorrect reasoning or factual error.
3. **Instruction Failure:** Violating explicit negative constraints, word limits, or required keywords.
4. **Format Failure:** Generating invalid JSON or incorrect bullet point counts.
5. **Verbosity:** Generating excessive conversational preamble or repetitive filler.
6. **Irrelevant:** Output that deviates entirely from the user prompt.
7. **Coding Failure:** Syntax errors (AST parsing failure) or failed unit tests.
8. **Safety Failure:** Failing to refuse malicious attack instructions.

---

### Production Deployment & CI/CD Harness Design

#### Q27: How should an LLM evaluation harness be integrated into a CI/CD pipeline?
**Answer:**
1. **Pre-Merge PR Smoke Test:** Run an automated fast-tier battery (50 core prompts, unit tests, syntax checks, safety guardrails) taking $< 60$ seconds. Block merges if safety compliance drops below 95% or if coding pass rate regresses.
2. **Nightly Regression Suite:** Run the full 200+ multi-category benchmark across all checkpoint variants.
3. **Canary / Shadow Evaluation:** Route 1% of production traffic through new model checkpoints, recording output comparisons and human feedback metrics.

#### Q28: How does temperature and sampling parameter choice affect evaluation reproducibility?
**Answer:** Non-zero temperature ($T > 0$) with top-$p$ or top-$k$ sampling introduces stochastic variance into generated tokens. Two runs of the exact same harness on the exact same checkpoint can yield different scores. For reproducible, deterministic benchmarking, standard practice uses **greedy decoding** ($T = 0.0$, `do_sample=False`). For evaluating creative generation or pass@$k$ metrics, multiple samples are generated per prompt with fixed random seeds.

#### Q29: What is the pass@$k$ metric, and how is it calculated without combinatorial bias?
**Answer:** In coding benchmarks (like HumanEval), pass@$k$ measures the probability that at least one of $k$ generated samples passes all test cases. Rather than evaluating all combinations naively, Chen et al. derived an unbiased estimator:
$$\text{pass@}k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}$$
where $n$ is total samples generated per prompt ($n \ge k$) and $c$ is the number of correct samples that passed all unit tests.

#### Q30: What is the difference between static benchmarks and dynamic / adaptive benchmarks?
**Answer:**
- **Static Benchmarks** (e.g., MMLU, GSM8K) have fixed question-answer pairs. Over time, web scraping and fine-tuning lead to benchmark saturation and data contamination.
- **Dynamic / Adaptive Benchmarks** (e.g., LMSYS Chatbot Arena, Dynabench) continuously generate new adversarial prompts, source live user conversations, or adapt question difficulty dynamically based on model skill level, ensuring lasting evaluative validity.
