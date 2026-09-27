# GRU SMS Spam Classification & Recurrent Model Benchmark

## Day 108 / 200

## Overview
A controlled benchmark comparing recurrent neural architectures for SMS spam classification.

## Models
- Simple RNN
- LSTM
- GRU
- Bidirectional GRU
- Stacked GRU

## Objective
Study the relationship between:
- model architecture
- parameter count
- training time
- classification performance

## Architecture
```text
Embedding
    ↓
   GRU
    ↓
 Dropout
    ↓
  Dense
    ↓
 Sigmoid
```

## Evaluation
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Average Precision

## Experiments
- **Exp 1:** Architecture Core Comparison (SimpleRNN vs LSTM vs GRU)
- **Exp 2:** Hidden units sweep (32, 64, 128) across RNN, LSTM, and GRU
- **Exp 3:** Sequence length sweep (20, 40, 60, 100)
- **Exp 4:** Dropout sweep (0.0, 0.2, 0.3, 0.5)
- **Exp 5:** Bidirectional GRU vs BiLSTM
- **Exp 6:** Stacked GRU vs Single-layer GRU

## Reproducibility
All models use the same dataset split, preprocessing pipeline, and random seed (`42`).
Executed natively in PyTorch on Python 3.14.4.

## Key Concepts
- Update gate
- Reset gate
- Hidden state
- Gradient flow
- Sequential modeling
- Recurrent architectures

---

## 💡 25 Interview Questions & Answers

### Beginner
1. **What is a GRU?**  
   A Gated Recurrent Unit is a specialized recurrent neural network architecture designed to capture sequential dependencies without the complex multi-gate structure of an LSTM.
2. **Why was GRU introduced?**  
   To resolve the vanishing gradient problem in vanilla RNNs while reducing the computational overhead and parameter footprint of LSTMs.
3. **How many gates does a GRU have?**  
   Two gates: the Update Gate ($z_t$) and the Reset Gate ($r_t$).
4. **What is the update gate?**  
   The gate that determines how much of the previous hidden state should be carried over into the new hidden state versus how much to replace with the candidate state.
5. **What is the reset gate?**  
   The gate that controls how much of the past hidden state should be remembered when calculating the new candidate state.
6. **Does GRU have a cell state?**  
   No. Unlike LSTM which separates cell state ($C_t$) and hidden state ($h_t$), GRU relies solely on the hidden state ($h_t$).
7. **What is the hidden state?**  
   The continuous vector representation that carries temporal memory and contextual representations across sequence time-steps.

### Intermediate
8. **Explain the GRU equations.**  
   - Update gate: $z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$  
   - Reset gate: $r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$  
   - Candidate state: $\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$  
   - New hidden state: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
9. **How does the reset gate work?**  
   When $r_t \approx 0$, the network acts as if it is reading the first symbol of an input sequence, allowing it to discard previously accumulated historical context.
10. **How does the update gate work?**  
    It acts as both a forget gate and an input gate combined: $z_t$ weights the new candidate while $(1 - z_t)$ weights the past hidden state.
11. **Why does GRU use sigmoid for gates?**  
    Sigmoid bounds values in $[0, 1]$, acting as a smooth probabilistic filter representing "what percentage of information to pass".
12. **Why does GRU use tanh for candidate activation?**  
    Tanh outputs values in $[-1, 1]$, allowing both positive and negative updates while maintaining zero-centered activations that prevent activation saturation.
13. **Why does GRU generally have fewer parameters than LSTM?**  
    GRU has 3 sets of transformations (Update, Reset, Candidate) while LSTM has 4 sets (Forget, Input, Candidate, Output), yielding approximately 25% fewer recurrent weights.
14. **What is `return_sequences`?**  
    A configuration setting determining whether the recurrent layer outputs the hidden state at every time-step $(T, H)$ or only the final hidden state $(H,)$.
15. **Why is `return_sequences=True` needed for stacked GRUs?**  
    Because subsequent recurrent layers expect a 3D sequential input $(B, T, H)$ to process through time, rather than a 2D summary vector.

### Advanced
16. **GRU vs LSTM: Which is better?**  
    Neither is universally superior. GRU is faster, consumes less memory, and excels on smaller datasets; LSTM can model more complex, long-horizon relationships due to its decoupled cell state.
17. **GRU vs vanilla RNN: Why is GRU superior?**  
    The linear interpolation in $h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$ creates shortcut gradient highways where error signals can backpropagate without exponential decay.
18. **Why can GRU handle long-term dependencies better than vanilla RNN?**  
    When $z_t \approx 0$, $h_t \approx h_{t-1}$, enabling the state to persist across dozens of steps without modification or gradient decay.
19. **Why might LSTM outperform GRU on certain tasks?**  
    LSTMs maintain independent cell and hidden states with distinct forget and output gates, allowing finer control over when memory is written, remembered, and exposed.
20. **Why does GRU train faster?**  
    It performs one fewer matrix multiplication per time step and requires fewer backpropagation gradient updates.
21. **What is a Bidirectional GRU?**  
    Two independent GRUs running in parallel—one forward and one backward across the sequence—concatenating their hidden states to capture past and future context.
22. **What is a Stacked GRU?**  
    Multiple GRU layers connected in series, allowing higher layers to learn abstract representations of lower-layer sequential features.
23. **Why can a classical TF-IDF model still outperform a recurrent neural network?**  
    On short, keyword-dense tasks (like spam), n-gram word presence alone is often sufficient; recurrent models require large datasets to avoid overfitting.
24. **Why is parameter count not enough to judge a model?**  
    Inference latency, memory bandwidth, gradient flow stability, and dataset size dictate real-world suitability far more than raw parameter counts.
25. **Why should model comparisons use the exact same data split and preprocessing?**  
    To eliminate confounding variables, guaranteeing that performance differentials arise strictly from architectural properties and not data leakage or variance.
