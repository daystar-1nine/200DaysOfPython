# 🚀 DAY 67 / 200 — Probability Distributions: Theoretical Foundations, Stochastic Modeling & SciPy Architecture

Welcome to **Day 67** of the 200 Days of Python Challenge.

Yesterday you mastered fundamental probability:
> **"Probability tells us how likely a single event or combination of events is to occur."**

Today, we take the crucial step forward into **Probability Distributions**:
> **"A probability distribution mathematically maps the entire landscape of possible outcomes to their corresponding probabilities or likelihood densities."**

Probability distributions form the mathematical backbone of modern Machine Learning, Statistical Inference, Hypothesis Testing, A/B Testing, Risk Modeling, Financial Engineering, and Generative AI.

---

# 📑 TABLE OF CONTENTS
1. [What is a Probability Distribution?](#1-what-is-a-probability-distribution)
2. [Discrete vs Continuous Random Variables](#2-discrete-vs-continuous-random-variables)
3. [The Core Four: PMF, PDF, CDF, and PPF](#3-the-core-four-pmf-pdf-cdf-and-ppf)
4. [Bernoulli Distribution](#4-bernoulli-distribution)
5. [Binomial Distribution](#5-binomial-distribution)
6. [Continuous Uniform Distribution](#6-continuous-uniform-distribution)
7. [Normal (Gaussian) Distribution](#7-normal-gaussian-distribution)
8. [Poisson Distribution](#8-poisson-distribution)
9. [Binomial vs Poisson vs Normal Approximations](#9-binomial-vs-poisson-vs-normal-approximations)
10. [Distribution Selection Decision Matrix](#10-distribution-selection-decision-matrix)
11. [Working with SciPy Stats: The Python API](#11-working-with-scipy-stats-the-python-api)
12. [Theoretical Probability vs Empirical Simulation](#12-theoretical-probability-vs-empirical-simulation)
13. [30 Technical Interview Questions & Answers](#13-30-technical-interview-questions--answers)
14. [Day 67 Assessment: Fully Worked Solutions](#14-day-67-assessment-fully-worked-solutions)
15. [10 Probability Distribution Insights for Practitioners](#15-10-probability-distribution-insights-for-practitioners)

---

# 1. WHAT IS A PROBABILITY DISTRIBUTION?

A **random variable** $X$ is a variable whose values depend on outcomes of a random phenomenon.
A **probability distribution** is a mathematical function that assigns probabilities to all possible values of $X$.

```text
Sample Space (S)  ──────>  Random Variable X(ω)  ──────>  Probability Law P(X)
All outcomes               Numerical Mapping              Assigned Weights
```

### Core Axiomatic Requirements:
1. **Non-negativity:** Every probability or density must be non-negative: $P(X = x) \ge 0$ or $f(x) \ge 0$.
2. **Total Probability:** The sum (for discrete) or integral (for continuous) over the entire support must equal 1:
   $$\sum_{x \in S} P(X = x) = 1 \quad \text{or} \quad \int_{-\infty}^{\infty} f(x)\,dx = 1$$

---

# 2. DISCRETE VS CONTINUOUS RANDOM VARIABLES

| Dimension | Discrete Random Variable | Continuous Random Variable |
| :--- | :--- | :--- |
| **Possible Values** | Countable (finite or countably infinite: $0, 1, 2, \dots$) | Uncountable continuum (intervals: $[a, b]$, $\mathbb{R}$) |
| **Probability Function** | **PMF** (Probability Mass Function) $P(X = x)$ | **PDF** (Probability Density Function) $f(x)$ |
| **Exact Value Probability** | $P(X = x) \ge 0$ (meaningful exact point probability) | $P(X = x) = 0$ (probability at a single point is zero) |
| **Interval Probability** | $\sum_{x=a}^b P(X=x)$ | $\int_a^b f(x)\,dx$ (Area under curve) |
| **Cumulative Function** | Step function $F(x) = \sum_{k \le x} P(X=k)$ | Continuous curve $F(x) = \int_{-\infty}^x f(t)\,dt$ |
| **Real-World Examples** | Number of purchases, defects, website clicks, coin flips | Heights, customer latency, transaction amount, temperature |
| **Key Distributions** | Bernoulli, Binomial, Poisson, Geometric, Hypergeometric | Uniform, Normal, Exponential, Beta, Gamma, Student's t |

---

# 3. THE CORE FOUR: PMF, PDF, CDF, AND PPF

Understanding these four mathematical functions is essential for every Data Scientist.

### A. PMF — Probability Mass Function (Discrete)
For discrete $X$, $p(x) = P(X = x)$.
- Gives the exact probability of observing value $x$.
- Range: $0 \le p(x) \le 1$.
- Sum: $\sum p(x) = 1$.

### B. PDF — Probability Density Function (Continuous)
For continuous $X$, $f(x)$ represents the relative likelihood of $X$ taking values near $x$.
- **Crucial:** $f(x)$ is **NOT** a probability! It is a **density** (probability per unit of $x$).
- $f(x)$ can be greater than 1! For example, if $X \sim \text{Uniform}(0, 0.2)$, then $f(x) = \frac{1}{0.2} = 5.0$.
- Single point probability is zero: $P(X = c) = \int_c^c f(x)\,dx = 0$.
- Probability over an interval $[a, b]$ is the area under $f(x)$:
  $$P(a \le X \le b) = \int_a^b f(x)\,dx$$

### C. CDF — Cumulative Distribution Function (Both Discrete & Continuous)
Answers: *"What is the probability that $X$ is less than or equal to $x$?"*
$$F(x) = P(X \le x)$$
- Monotonically non-decreasing: if $x_1 < x_2$, then $F(x_1) \le F(x_2)$.
- Limits: $\lim_{x 	o -\infty} F(x) = 0$ and $\lim_{x 	o \infty} F(x) = 1$.
- Complement Rule: $P(X > x) = 1 - F(x)$.
- Interval Rule: $P(a < X \le b) = F(b) - F(a)$.

### D. PPF — Percent Point Function / Quantile Function (Inverse CDF)
Answers: *"What value $x$ has a cumulative probability of $q$?"*
$$x = F^{-1}(q) \iff P(X \le x) = q$$
- In SciPy: `distribution.ppf(q)`.
- Used to compute median ($q = 0.5$), quartiles (Q1: $q = 0.25$, Q3: $q = 0.75$), value-at-risk, and critical values in hypothesis testing.

```text
    Value x  ────────── CDF F(x) ──────────>  Probability q (0 to 1)
    Value x  <───────── PPF F⁻¹(q) ─────────  Probability q (0 to 1)
```

---

# 4. BERNOULLI DISTRIBUTION

A Bernoulli trial is an experiment with exactly two mutually exclusive outcomes: Success ($X = 1$) or Failure ($X = 0$).

### Parameters & Formulas
- **Parameter:** $p \in [0, 1]$ (Probability of success).
- **PMF:**
  $$P(X = x) = p^x (1 - p)^{1 - x}, \quad x \in \{0, 1\}$$
- **Expected Value:** $E[X] = 1 \cdot p + 0 \cdot (1 - p) = p$
- **Variance:**
  $$\text{Var}(X) = E[X^2] - (E[X])^2 = (1^2 \cdot p + 0^2 \cdot (1-p)) - p^2 = p - p^2 = p(1 - p)$$
- **Standard Deviation:** $\sigma = \sqrt{p(1 - p)}$

### Python Example
```python
from scipy.stats import bernoulli

p = 0.7
rv = bernoulli(p)

print("P(X = 1):", rv.pmf(1))       # 0.7
print("P(X = 0):", rv.pmf(0))       # 0.3
print("Mean:", rv.mean())           # 0.7
print("Variance:", rv.var())        # 0.21
```

---

# 5. BINOMIAL DISTRIBUTION

The Binomial distribution models the number of successes $k$ in $n$ independent and identical Bernoulli trials.

### Four Required Assumptions (BINS):
1. **B**inary: Each trial has exactly two outcomes (success or failure).
2. **I**ndependent: The outcome of any trial does not influence any other trial.
3. **N**umber of trials is fixed in advance ($n$).
4. **S**ame probability of success on each trial ($p$).

### Mathematical Formulation
- **Parameters:** $n \in \mathbb{N}$ (number of trials), $p \in [0, 1]$ (success probability).
- **PMF:**
  $$P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k} = \frac{n!}{k!(n - k)!} p^k (1 - p)^{n - k}, \quad k \in \{0, 1, \dots, n\}$$
- **Expected Value:** $E[X] = np$
- **Variance:** $\text{Var}(X) = np(1 - p)$
- **Standard Deviation:** $\sigma = \sqrt{np(1 - p)}$

### Complement Trick for "At Least $k$ Successes":
$$P(X \ge k) = 1 - P(X \le k - 1) = 1 - \text{CDF}(k - 1)$$

### Python Example
```python
from scipy.stats import binom

n, p = 20, 0.1
rv = binom(n, p)

p_exact_3 = rv.pmf(3)                 # P(X = 3)
p_at_most_3 = rv.cdf(3)               # P(X <= 3)
p_at_least_3 = 1 - rv.cdf(2)          # P(X >= 3)
ev = rv.mean()                        # 20 * 0.1 = 2.0
variance = rv.var()                   # 20 * 0.1 * 0.9 = 1.8
```

---

# 6. CONTINUOUS UNIFORM DISTRIBUTION

A continuous random variable $X$ has a Uniform distribution over $[a, b]$ if all sub-intervals of equal length within $[a, b]$ are equally likely.

### Mathematical Formulation
- **Parameters:** $a, b \in \mathbb{R}$ with $a < b$.
- **PDF:**
  $$f(x) = \begin{cases} \frac{1}{b - a}, & a \le x \le b \ 0, & \text{otherwise} \end{cases}$$
- **CDF:**
  $$F(x) = \begin{cases} 0, & x < a \ \frac{x - a}{b - a}, & a \le x \le b \ 1, & x > b \end{cases}$$
- **Expected Value:** $E[X] = \frac{a + b}{2}$
- **Variance:** $\text{Var}(X) = \frac{(b - a)^2}{12}$

### Python Example
```python
from scipy.stats import uniform

# SciPy parameterization: loc = a, scale = (b - a)
a, b = 0, 10
rv = uniform(loc=a, scale=b - a)

pdf_val = rv.pdf(5.0)    # 1 / 10 = 0.1
cdf_val = rv.cdf(7.0)    # (7 - 0) / 10 = 0.7
```

---

# 7. NORMAL (GAUSSIAN) DISTRIBUTION

The **Normal distribution** is the cornerstone of probability and statistics due to the Central Limit Theorem.

### Mathematical Formulation
- **Parameters:** $\mu \in \mathbb{R}$ (mean / center), $\sigma > 0$ (standard deviation / spread).
- **Notation:** $X \sim \mathcal{N}(\mu, \sigma^2)$
- **PDF:**
  $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}
ight)$$
- **Expected Value:** $E[X] = \mu$
- **Variance:** $\text{Var}(X) = \sigma^2$
- **Symmetry:** $\text{Mean} = \text{Median} = \text{Mode} = \mu$. Skewness $= 0$, Excess Kurtosis $= 0$.

### Standard Normal Distribution
A special case where $\\mu = 0$ and $\sigma = 1$, denoted by $Z \sim \mathcal{N}(0, 1)$.
Any normal variable can be standardized via the Z-score transformation:
$$Z = \frac{X - \mu}{\sigma} \iff X = \mu + Z\sigma$$

### The Empirical Rule (68–95–99.7% Rule):
- $P(\mu - 1\sigma \le X \le \mu + 1\sigma) \approx 68.27\%$
- $P(\mu - 2\sigma \le X \le \mu + 2\sigma) \approx 95.45\%$
- $P(\mu - 3\sigma \le X \le \mu + 3\sigma) \approx 99.73\%$

### Effects of Parameter Shifts:
- **Changing $\mu$:** Translates the distribution along the horizontal axis without altering its width or peak height.
- **Increasing $\sigma$:** Flattens and broadens the bell curve, keeping the total area equal to 1.
- **Decreasing $\sigma$:** Concentrates density tightly around $\mu$, making the peak taller.

---

# 8. POISSON DISTRIBUTION

The Poisson distribution models the number of events occurring in a fixed interval of time or space, given that these events occur independently and at a constant average rate $\lambda$.

### Assumptions:
1. Events occur one at a time (no simultaneous events).
2. The rate $\lambda$ is constant over the interval.
3. Occurrences in non-overlapping intervals are independent.

### Mathematical Formulation
- **Parameter:** $\lambda > 0$ (expected arrival rate per interval).
- **PMF:**
  $$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k \in \{0, 1, 2, \dots\}$$
- **Expected Value:** $E[X] = \lambda$
- **Variance:** $\text{Var}(X) = \lambda$
- **Equidispersion:** A distinctive property of Poisson is that $\text{Mean} = \text{Variance} = \lambda$.

### Real-World Applications:
- Server requests per second.
- Support calls arriving per hour.
- Physical manufacturing defects per square meter of fabric.
- Hospital emergency room admissions per day.

---

# 9. BINOMIAL VS POISSON VS NORMAL APPROXIMATIONS

### A. Poisson Approximation to the Binomial (Law of Rare Events)
When $n$ is very large and $p$ is very small such that $\lambda = np$ remains moderate ($n \ge 20, p \le 0.05$):
$$\binom{n}{k} p^k (1 - p)^{n - k} \approx \frac{e^{-\lambda}\lambda^k}{k!}$$
*Why use it?* Evaluating $\binom{1000}{3} (0.002)^3 (0.998)^{997}$ directly is computationally heavy; $\frac{e^{-2} 2^3}{3!}$ is immediate and practically identical.

### B. Normal Approximation to the Binomial (De Moivre–Laplace Theorem)
When $np \ge 10$ and $n(1 - p) \ge 10$, the Binomial distribution $\text{Binom}(n, p)$ converges to a continuous Normal distribution:
$$$\text{Binom}(n, p) \approx \mathcal{N}\big(\\mu = np, \, \sigma^2 = np(1 - p)\big)$$$
*Continuity Correction:* Since discrete integers are approximated by continuous intervals, evaluate $P(X \le k)$ as $P(X_{\text{norm}} \le k + 0.5)$.

---

# 10. DISTRIBUTION SELECTION DECISION MATRIX

| Problem Scenario | Nature of Variable | Appropriate Distribution | Key Parameters |
| :--- | :--- | :--- | :--- |
| Single binary trial (conversion/churn) | Discrete (0 or 1) | **Bernoulli** | $p$ (success probability) |
| Success count in fixed trials $n$ | Discrete ($0 \le k \le n$) | **Binomial** | $n$ (trials), $p$ (probability) |
| Event arrivals over continuous time/space | Discrete count ($k \ge 0$) | **Poisson** | $\lambda$ (average rate) |
| Continuous measurement equally likely in range | Continuous ($a \le x \le b$) | **Uniform** | $a$ (min), $b$ (max) |
| Bell-shaped natural or measurement data | Continuous ($-\infty < x < \infty$) | **Normal** | $\mu$ (mean), $\sigma$ (std dev) |

---

# 11. WORKING WITH SCIPY STATS: THE PYTHON API

SciPy provides unified distribution objects in `scipy.stats`.

| Method | Description | Example Syntax |
| :--- | :--- | :--- |
| `pmf(k, ...)` | Discrete probability mass $P(X = k)$ | `binom.pmf(k=3, n=10, p=0.5)` |
| `pdf(x, ...)` | Continuous probability density $f(x)$ | `norm.pdf(x=75, loc=70, scale=10)` |
| `cdf(x, ...)` | Cumulative probability $P(X \le x)$ | `poisson.cdf(k=4, mu=5.0)` |
| `sf(x, ...)` | Survival function $P(X > x) = 1 - \text{CDF}(x)$ | `norm.sf(x=80, loc=70, scale=10)` |
| `ppf(q, ...)` | Percent point function (quantile) $F^{-1}(q)$ | `norm.ppf(q=0.95, loc=70, scale=10)` |
| `rvs(size, ...)` | Generate random variates / samples | `norm.rvs(loc=70, scale=10, size=1000)` |
| `mean(...)` | Theoretical distribution mean | `binom.mean(n=20, p=0.4)` |
| `var(...)` | Theoretical distribution variance | `poisson.var(mu=6.0)` |
| `std(...)` | Theoretical distribution standard deviation | `norm.std(scale=15.0)` |

---

# 12. THEORETICAL PROBABILITY VS EMPIRICAL SIMULATION

Why doesn't a simulated sample of 10,000 rolls exactly match theoretical predictions?

1. **Stochastic Variation:** Random sampling is subject to finite-sample noise. Every simulated draw is an independent realization.
2. **Law of Large Numbers (LLN):** As sample size $N 	o \infty$, the sample mean $\bar{X}_N$ converges almost surely to the theoretical expected value $E[X]$.
3. **Standard Error of the Mean:** The expected magnitude of simulation error shrinks as:
   $$\text{SE} = \frac{\sigma}{\sqrt{N}}$$
   To reduce simulation error by a factor of 10, one must generate $10^2 = 100	imes$ more samples!

---

# 13. 30 TECHNICAL INTERVIEW QUESTIONS & ANSWERS

### Probability Distributions
#### 1. What is a probability distribution?
A probability distribution is a mathematical function that describes the likelihood of obtaining the possible values that a random variable can assume. It completely specifies the probabilities of all subsets of outcomes in the sample space.

#### 2. What is the fundamental difference between discrete and continuous distributions?
In discrete distributions, the random variable takes on countable values, and probabilities are assigned directly to individual points via a Probability Mass Function (PMF). In continuous distributions, the variable takes on uncountable values across an interval, probability at any single exact point is zero, and probabilities are assigned to intervals via integration of a Probability Density Function (PDF).

#### 3. What is a PMF?
A Probability Mass Function (PMF) maps each discrete value $x$ of a discrete random variable $X$ to its exact probability: $p(x) = P(X = x)$, where $0 \le p(x) \le 1$ and $\sum_x p(x) = 1$.

#### 4. What is a PDF?
A Probability Density Function (PDF) describes the relative likelihood for a continuous random variable to take on a given value. The probability of landing in an interval $[a, b]$ is the definite integral of the PDF over that interval: $P(a \le X \le b) = \int_a^b f(x)\,dx$.

#### 5. What is a CDF?
The Cumulative Distribution Function (CDF) gives the probability that a random variable $X$ takes a value less than or equal to $x$: $F(x) = P(X \le x)$. It applies to both discrete and continuous variables, is monotonically non-decreasing, and ranges from 0 to 1.

#### 6. What is a PPF?
The Percent Point Function (PPF), also known as the Quantile Function or Inverse CDF, takes a cumulative probability $q \in [0, 1]$ and returns the corresponding threshold value $x$ such that $P(X \le x) = q$.

#### 7. What is the key difference between PMF and PDF?
PMF values represent true probabilities ($0 \le P(X=x) \le 1$). PDF values represent density (probability per unit length) and can exceed 1. A PDF value must be integrated over an interval to yield a probability.

#### 8. Can a PDF value be greater than 1?
Yes! For continuous distributions with a narrow support, $f(x)$ can exceed 1. For instance, for $X \sim \text{Uniform}(0, 0.5)$, $f(x) = \frac{1}{0.5 - 0} = 2.0$ for all $x \in [0, 0.5]$. The only requirement is that the total integral $\int f(x)\,dx = 1$.

#### 9. What does the area under a PDF curve represent?
The area under a PDF curve between two points $a$ and $b$ represents the probability that the random variable falls within that interval: $P(a \le X \le b)$. The total area under the entire curve across $(-\infty, \infty)$ is strictly 1.0.

---

### Bernoulli & Binomial Distributions
#### 10. What is a Bernoulli distribution?
A Bernoulli distribution is the probability distribution of a single random experiment that has exactly two possible outcomes: Success ($X = 1$) with probability $p$, and Failure ($X = 0$) with probability $1 - p$.

#### 11. What is a Binomial distribution?
A Binomial distribution models the total number of successes observed across $n$ independent and identically distributed Bernoulli trials, each having a constant success probability $p$.

#### 12. What four assumptions are required for a Binomial distribution?
1. Fixed number of trials $n$.
2. Only two outcomes per trial (binary).
3. Constant probability of success $p$ on every trial.
4. Independent trials (the result of one does not affect another).

#### 13. How are the Bernoulli and Binomial distributions related?
The Bernoulli distribution is a special case of the Binomial distribution where $n = 1$. Conversely, a Binomial random variable is the sum of $n$ independent Bernoulli random variables: $X = \sum_{i=1}^n Y_i$, where $Y_i \sim \text{Bernoulli}(p)$.

#### 14. What are the formulas for the mean and variance of a Binomial distribution?
- Mean: $E[X] = np$
- Variance: $\text{Var}(X) = np(1 - p)$
- Standard Deviation: $\sigma = \sqrt{np(1 - p)}$

---

### Normal Distribution
#### 15. What is a Normal distribution?
A Normal (or Gaussian) distribution is a continuous, symmetric, bell-shaped probability distribution defined on $\mathbb{R}$ whose density peaks at the mean and drops exponentially with the square of the distance from the mean.

#### 16. What parameters define a Normal distribution?
- Mean ($\mu$): The location parameter, determining the center of symmetry.
- Standard Deviation ($\sigma$): The scale parameter ($\sigma > 0$), determining the spread or dispersion.

#### 17. What is a Standard Normal distribution?
A Standard Normal distribution is a Normal distribution with $\\mu = 0$ and $\sigma = 1$, denoted $Z \sim \mathcal{N}(0, 1)$. Any normal variable can be transformed into standard normal via $Z = (X - \mu) / \sigma$.

#### 18. What is the 68–95–99.7 Empirical Rule?
In a normal distribution:
- Approximately 68.27% of observations fall within $\mu \pm 1\sigma$.
- Approximately 95.45% of observations fall within $\mu \pm 2\sigma$.
- Approximately 99.73% of observations fall within $\mu \pm 3\sigma$.

#### 19. What happens to the normal curve when standard deviation increases?
As $\sigma$ increases, the curve spreads wider along the horizontal axis, and the peak height decreases so that the total area remains exactly 1.

#### 20. What happens to the normal curve when the mean changes?
Changing $\mu$ translates (shifts) the entire bell curve horizontally along the x-axis without changing its shape, width, or height.

---

### Poisson Distribution
#### 21. What is a Poisson distribution?
A Poisson distribution is a discrete probability distribution that expresses the probability of a given number of independent events occurring in a fixed interval of time or space, given a known constant average rate $\lambda$.

#### 22. What does $\lambda$ represent in a Poisson distribution?
$\lambda$ (lambda) represents the expected (average) number of event occurrences within the specified interval.

#### 23. Why are the mean and variance equal in a Poisson distribution?
In a Poisson process with rate $\lambda$, mathematically $E[X] = \sum k \frac{e^{-\lambda}\lambda^k}{k!} = \lambda$ and $E[X(X-1)] = \lambda^2$, yielding $\text{Var}(X) = E[X^2] - (E[X])^2 = (\lambda^2 + \lambda) - \lambda^2 = \lambda$. This property is called **equidispersion**.

#### 24. What is the key distinction between Binomial and Poisson distributions?
Binomial has a fixed number of trials $n$ and counts successes out of $n$ ($0 \le k \le n$). Poisson has no fixed number of trials; it counts events over a continuous interval of time/space ($k \in \{0, 1, 2, \dots, \infty\}$).

#### 25. When can a Poisson distribution approximate a Binomial distribution?
When $n$ is large ($n \ge 20$) and $p$ is small ($p \le 0.05$) such that $np$ is moderate, a $\text{Binomial}(n, p)$ distribution is closely approximated by a $\text{Poisson}(\lambda = np)$.

---

### Python & SciPy Mechanics
#### 26. What does `scipy.stats` provide for distributions?
`scipy.stats` provides continuous and discrete distribution classes with unified methods for evaluating probability mass/density (`pmf`/`pdf`), cumulative probabilities (`cdf`), survival probabilities (`sf`), quantiles (`ppf`), random generation (`rvs`), and moments (`mean`, `var`, `std`).

#### 27. What is the difference between `pmf()` and `cdf()` in SciPy?
`pmf(k)` computes point probability $P(X = k)$, while `cdf(k)` computes cumulative probability $P(X \le k) = \sum_{i \le k} P(X = i)$.

#### 28. What does `ppf()` do?
`ppf(q)` computes the inverse of the CDF (Quantile function). It returns the value $x$ such that the probability of $X \le x$ is equal to $q$.

#### 29. What does `rvs()` do?
`rvs(size, random_state)` generates random variates (Monte Carlo samples) drawn from the specified distribution.

#### 30. Why should stochastic simulations use a fixed seed during testing?
Pseudo-random number generators produce deterministic sequences from a seed. Setting `random_state` or `seed` ensures reproducible test runs, eliminating flaky tests caused by random sampling fluctuations.

---

# 14. DAY 67 ASSESSMENT: FULLY WORKED SOLUTIONS

### PART 1 — Bernoulli
A customer has a 30% probability of purchasing ($p = 0.3$).
- **$P(\text{Purchase}) = P(X = 1) = p = 0.30$**
- **$P(\text{No Purchase}) = P(X = 0) = 1 - p = 0.70$**
- **$\text{Mean } E[X] = p = 0.30$**
- **$\text{Variance } \text{Var}(X) = p(1 - p) = 0.30 	imes 0.70 = 0.21$**

---

### PART 2 — Binomial
$n = 100$ customers, purchase probability $p = 0.05$.
- **$P(X = 5)$:**
  $$P(X = 5) = \binom{100}{5} (0.05)^5 (0.95)^{95} \approx 0.1800178 \approx 18.00\%$$
- **$P(X \le 5)$:**
  $$P(X \le 5) = \sum_{k=0}^5 \binom{100}{k} (0.05)^k (0.95)^{100-k} \approx 0.615999 \approx 61.60\%$$
- **$P(X \ge 5)$:**
  $$P(X \ge 5) = 1 - P(X \le 4) \approx 1 - 0.435981 = 0.564019 \approx 56.40\%$$
- **$\text{Expected Purchases } E[X] = np = 100 	imes 0.05 = 5.0$**
- **$\text{Standard Deviation } \sigma = \sqrt{np(1 - p)} = \sqrt{100 	imes 0.05 	imes 0.95} = \sqrt{4.75} \approx 2.17945$**

---

### PART 3 — Normal
Exam scores: $X \sim \mathcal{N}(\\mu = 70, \sigma = 10)$.
- **$P(X < 60)$:**
  $$Z = \frac{60 - 70}{10} = -1.0 \implies P(Z < -1.0) = \Phi(-1.0) \approx 0.158655 \approx 15.87\%$$
- **$P(X > 80)$:**
  $$Z = \frac{80 - 70}{10} = +1.0 \implies P(Z > +1.0) = 1 - \Phi(1.0) \approx 0.158655 \approx 15.87\%$$
- **$P(60 < X < 80)$:**
  $$P(-1.0 < Z < 1.0) = \Phi(1.0) - \Phi(-1.0) \approx 0.841345 - 0.158655 = 0.682689 \approx 68.27\%$$

---

### PART 4 — Poisson
Average arrival rate: $\lambda = 8$ support calls/hour.
- **$P(X = 5)$:**
  $$P(X = 5) = \frac{e^{-8} 8^5}{5!} = \frac{0.00033546 	imes 32768}{120} \approx 0.091604 \approx 9.16\%$$
- **$P(X \le 5)$:**
  $$P(X \le 5) = \sum_{k=0}^5 \frac{e^{-8} 8^k}{k!} \approx 0.191236 \approx 19.12\%$$
- **$P(X > 8)$:**
  $$P(X > 8) = 1 - P(X \le 8) \approx 1 - 0.592547 = 0.407453 \approx 40.75\%$$

---

### PART 5 — Conceptual Question
**Question:** *Why does a continuous random variable generally have probability 0 at one exact point, yet still have meaningful probabilities over intervals?*

**Answer:**
A continuous random variable can take on any of an **uncountably infinite** number of values within any interval $[a, b]$. In classical probability, if we assigned any non-zero probability $\epsilon > 0$ to every point, the sum of probabilities over uncountably many points would diverge to infinity, violating the total probability axiom ($\int f(x)\,dx = 1$). 
Mathematically, the probability of an exact value $c$ is:
$$P(X = c) = \int_c^c f(x)\,dx = 0$$
However, probability is defined on sets (intervals) as a **measure** (area under the curve). Just as a geometric line has length while an individual point on the line has zero length, an interval has a positive probability measure while single points have measure zero.

---

### PART 6 — Practical Simulation
Generate $N = 100,000$ samples from $\mathcal{N}(\\mu = 70, \sigma = 10)$ with seed 42.
- **Theoretical Values:**
  - Mean: $70.0$
  - Median: $70.0$
  - Std: $10.0$
  - P10: $70 + 10 	imes \Phi^{-1}(0.10) \approx 70 - 1.28155 	imes 10 = 57.1845$
  - P90: $70 + 10 	imes \Phi^{-1}(0.90) \approx 70 + 1.28155 	imes 10 = 82.8155$
- **Simulated Metrics ($N = 100,000$):**
  - Simulated Mean: $\approx 70.005$
  - Simulated Median: $\approx 70.002$
  - Simulated Std: $\approx 9.992$
  - Simulated P10: $\approx 57.181$
  - Simulated P90: $\approx 82.823$
- **Why doesn't the simulation match theory exactly?**
  Finite random sampling introduces random sampling error. By the Central Limit Theorem and Law of Large Numbers, the error between the sample mean and true mean is bounded by the standard error:
  $$\text{SE} = \frac{\sigma}{\sqrt{N}} = \frac{10}{\sqrt{100,000}} \approx 0.0316$$
  The observed deviation ($|70.005 - 70.0| = 0.005$) is well within the expected $\pm 2\text{SE} \approx \pm 0.0632$ error envelope.

---

# 15. 10 PROBABILITY DISTRIBUTION INSIGHTS FOR PRACTITIONERS

1. **Check Distribution Assumptions First:** A model assuming normally distributed residuals will fail if data is heavy-tailed, zero-inflated, or strictly positive.
2. **Never Treat a PDF as a Probability:** A PDF value can exceed 1.0; always integrate over an interval or use `cdf()` differences.
3. **Use the Complement Trick:** For discrete inequalities ($P(X \ge k)$), always compute $1 - \text{CDF}(k - 1)$ to avoid slow, error-prone manual loops.
4. **Beware of Equidispersion in Poisson:** If real-world count data has variance significantly higher than its mean (overdispersion), replace Poisson with a **Negative Binomial** distribution.
5. **The Power of Standardizing:** Any normal problem becomes trivial when transformed to $Z = (X - \mu) / \sigma$.
6. **Quantiles are the Inverse of Probabilities:** Use `ppf()` to set SLA cutoffs, threshold limits, and Value-at-Risk targets.
7. **Small Probabilities in Large Systems (Poisson Limit):** High-traffic systems with rare individual events (clicks, crashes, fraudulent logins) are naturally Poisson distributed.
8. **Simulate to Validate Analytical Derivations:** When unsure of a complex probability derivation, run $10^5$ Monte Carlo trials to empirically check your math.
9. **Continuous Uniform is the Parent of All Distributions:** Using the **Inverse Transform Sampling** theorem, generating $U \sim \text{Uniform}(0, 1)$ allows sampling from any distribution via $X = F^{-1}(U)$.
10. **Error Shrinks with $1/\sqrt{N}$:** Increasing Monte Carlo accuracy tenfold requires 100 times more compute. Always plan simulation budgets accordingly.
