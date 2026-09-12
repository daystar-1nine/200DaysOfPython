# 🚀 DAY 70 / 200 — Hypothesis Testing Fundamentals

**Progress:** 70 / 200 → **35.0% complete**  
**Remaining:** **130 days**  
**Milestone:** 🎉 Over one-third of the 200-day journey completed!

---

## 🧭 1. Executive Summary & Why Hypothesis Testing Matters

In previous days, you learned how to compute point estimates (sample mean $\bar{x}$, sample proportion $\hat{p}$) and construct confidence intervals to quantify estimation uncertainty.

However, business and scientific problems rarely stop at estimation. Stakeholders don't merely ask: *"What is the mean delivery time?"*  
They ask decisive questions:
- *"Did the new routing algorithm reduce delivery times below our 30-minute SLA guarantee?"*
- *"Is the new landing page converting significantly better than our 10% baseline, or was the uptick pure random noise?"*
- *"Is our manufacturing line calibration drifting away from the required 500g specification?"*
- *"Does the new loyalty program increase average basket size above $120?"*

**Hypothesis testing** is the formalized mathematical engine for making decisions under uncertainty. It provides an objective decision rule to separate **genuine signal** from **stochastic sampling variation**.

---

## 🏛️ 2. Core Foundations & Theoretical Framework

### 2.1 The Philosophy of Hypothesis Testing
Hypothesis testing operates under the principle of **Popperian falsification**: we cannot strictly "prove" an affirmative scientific theory with finite observational data; instead, we formulate a conservative baseline hypothesis (**the null hypothesis**, $H_0$) and examine whether the empirical evidence is strong enough to reject it.

This mirrors the modern criminal legal system:
- **Null Hypothesis ($H_0$):** The defendant is innocent (no effect, no difference, status quo).
- **Alternative Hypothesis ($H_1$):** The defendant is guilty (an effect exists, difference is real).
- **Burden of Proof:** Innocent until proven guilty beyond a reasonable doubt. We never "accept" $H_0$ as proven fact; we either **reject $H_0$** or **fail to reject $H_0$**.

```
                           REALITY (TRUTH)
                    $H_0$ is TRUE        $H_0$ is FALSE
              ┌──────────────────────┬──────────────────────┐
Reject $H_0$  │    TYPE I ERROR      │   CORRECT DECISION   │
              │  (False Positive)    │   Statistical Power  │
DECISION      │    Probability = α   │   Probability = 1 - β│
              ├──────────────────────┼──────────────────────┤
Fail to       │   CORRECT DECISION   │    TYPE II ERROR     │
Reject $H_0$  │   Confidence Level   │  (False Negative)    │
              │    Probability = 1 - α│   Probability = β    │
              └──────────────────────┴──────────────────────┘
```

---

### 2.2 Formulating Hypotheses: Null ($H_0$) vs Alternative ($H_1$)

1. **The Null Hypothesis ($H_0$):**
   - The default, conservative claim of "no difference", "no change", or "no treatment effect".
   - Always contains an equality condition ($=$, $\le$, or $\ge$).
   - Examples:
     - $H_0: \mu = 500$g (machine is calibrated).
     - $H_0: \mu \le 30$ min (delivery meets SLA).
     - $H_0: p \le 0.10$ (conversion rate does not exceed 10%).

2. **The Alternative Hypothesis ($H_1$ or $H_a$):**
   - The research hypothesis or claim we suspect is true and seek evidence to support.
   - Always represents a strict inequality ($
e$, $>$, or $<$).
   - Examples:
     - $H_1: \mu \ne 500$g (machine calibration has shifted).
     - $H_1: \mu > 30$ min (delivery violates SLA).
     - $H_1: p > 0.10$ (conversion rate exceeds 10%).

---

### 2.3 One-Tailed vs Two-Tailed Tests

| Dimension | Two-Tailed Test (Non-Directional) | Right-Tailed Test (Directional) | Left-Tailed Test (Directional) |
| :--- | :--- | :--- | :--- |
| **Research Question** | Has the parameter changed in either direction? | Has the parameter increased? | Has the parameter decreased? |
| **Hypotheses** | $H_0: \mu = \mu_0$ vs $H_1: \mu \ne \mu_0$ | $H_0: \mu \le \mu_0$ vs $H_1: \mu > \mu_0$ | $H_0: \mu \ge \mu_0$ vs $H_1: \mu < \mu_0$ |
| **Rejection Region** | Split across both tails: $\alpha/2$ in each tail | Entire $\alpha$ in upper tail | Entire $\alpha$ in lower tail |
| **Critical Values** | $\pm z_{\alpha/2}$ or $\pm t_{\alpha/2, df}$ | $+z_{\alpha}$ or $+t_{\alpha, df}$ | $-z_{\alpha}$ or $-t_{\alpha, df}$ |
| **P-Value Calculation** | $2 \times P(Z \ge |z_{obs}|)$ | $P(Z \ge z_{obs})$ | $P(Z \le z_{obs})$ |
| **Statistical Power** | Lower power in a specific direction | Higher power for positive effects | Higher power for negative effects |

---

### 2.4 Significance Level ($\alpha$), Test Statistics, and P-Values

#### 1. Significance Level ($\alpha$)
- The pre-specified tolerance for committing a **Type I error** (rejecting a true null hypothesis).
- Standard scientific default: $\alpha = 0.05$ (5% risk of false positive).
- High-stakes contexts (clinical trials, aerospace engineering): $\alpha = 0.01$ or $\alpha = 0.001$.
- Early exploratory experiments: $\alpha = 0.10$.

#### 2. Test Statistics
A standardized metric quantifying how far the observed sample statistic deviates from the null parameter value, measured in standard error units:

$$\text{Test Statistic} = \frac{\text{Sample Statistic} - \text{Null Parameter Value}}{\text{Standard Error of the Statistic}}$$

- **1-Sample Z-Test for Mean (known population $\sigma$):**
  $$z = \frac{\bar{x} - \mu_0}{\frac{\sigma}{\sqrt{n}}} = \frac{(\bar{x} - \mu_0)\sqrt{n}}{\sigma}$$

- **1-Sample Student's t-Test for Mean (unknown population $\sigma$, estimated by sample $s$):**
  $$t = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}} = \frac{(\bar{x} - \mu_0)\sqrt{n}}{s}, \quad \text{with } df = n - 1$$

- **1-Sample Z-Test for Proportion ($p_0$ under $H_0$):**
  $$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1 - p_0)}{n}}}$$

#### 3. P-Value (Probability Value)
- **Precise Definition:** The probability, under the assumption that the null hypothesis $H_0$ is true, of observing a test statistic as extreme as, or more extreme than, the statistic actually observed from the sample.
- **The ASA Statement on Statistical Significance:**
  1. $P$-values can indicate how incompatible the data are with a specified statistical model.
  2. $P$-values do not measure the probability that the studied hypothesis is true ($P(H_0 | \text{data}) \ne p\text{-value}$).
  3. Scientific conclusions and business decisions should not be based solely on whether a $p$-value passes a specific threshold ($p < 0.05$).
  4. Proper inference requires full reporting and transparency.
  5. A $p$-value does not measure the size of an effect or the importance of a result.
  6. By itself, a $p$-value does not provide a good measure of evidence regarding a model or hypothesis.

---

### 2.5 Decision Rules

Two equivalent methodologies lead to the exact same conclusion:

#### Method A: The P-Value Decision Rule
- If $p\text{-value} \le \alpha$: **Reject $H_0$** (Statistical significance detected at level $\alpha$).
- If $p\text{-value} > \alpha$: **Fail to reject $H_0$** (Insufficient evidence to reject status quo).

#### Method B: The Critical Value Decision Rule
- Determine the critical value $c$ corresponding to $\alpha$ (e.g. for two-tailed $z$-test with $\alpha = 0.05$, $c = \pm 1.96$).
- If $|z_{obs}| \ge z_{critical}$: **Reject $H_0$** ($z_{obs}$ falls inside the rejection region).
- If $|z_{obs}| < z_{critical}$: **Fail to reject $H_0$**.

---

### 2.6 Confidence Interval Duality with Two-Tailed Hypothesis Tests

There is a fundamental mathematical equivalence between a two-tailed hypothesis test at significance level $\alpha$ and a $(1 - \alpha)$ confidence interval:
- If the null hypothesized value $\mu_0$ falls **inside** the $(1 - \alpha)$ confidence interval:
  $$\mu_0 \in [\bar{x} - ME, \; \bar{x} + ME] \iff p\text{-value} > \alpha \iff \text{Fail to reject } H_0$$
- If the null hypothesized value $\mu_0$ falls **outside** the $(1 - \alpha)$ confidence interval:
  $$\mu_0 \notin [\bar{x} - ME, \; \bar{x} + ME] \iff p\text{-value} \le \alpha \iff \text{Reject } H_0$$

Confidence intervals are strictly superior for business decision-making because they communicate both the **test decision** and the **magnitude/uncertainty of the effect**.

---

### 2.7 Statistical Significance vs Practical Significance & Effect Size

#### Why Statistical Significance Can Mislead
With very large sample sizes ($n = 100,000$), the standard error $\frac{s}{\sqrt{n}} \to 0$. Consequently, trivial, negligible deviations (e.g., an increase in delivery time of 0.02 seconds) achieve $p < 0.0001$. This is **statistically significant**, but practically worthless.

#### Cohen's d (Standardized Effect Size)
To evaluate practical significance independently of sample size $n$, we compute Cohen's $d$:

$$d = \frac{\bar{x} - \mu_0}{s}$$

| Absolute Cohen's $d$ | Effect Size Magnitude | Practical Interpretation |
| :--- | :--- | :--- |
| $|d| < 0.20$ | Negligible | Negligible practical difference; barely perceptible |
| $0.20 \le |d| < 0.50$ | Small | Noticeable only through careful statistical measurement |
| $0.50 \le |d| < 0.80$ | Medium | Substantial effect observable in standard operations |
| $|d| \ge 0.80$ | Large | Massive effect obvious to everyday observers |

---

## 🏭 3. Real-World Business Case Studies

### Case 1: E-Commerce Delivery SLA Verification
- **Context:** An instant-grocery delivery startup advertises a guarantee: "Average delivery time is at most 30 minutes".
- **Data:** $n = 250$ delivery runs.
- **Sample Metrics:** $\bar{x} = 31.84$ mins, $s = 5.48$ mins.
- **Hypotheses:** $H_0: \mu \le 30.0$ vs $H_1: \mu > 30.0$ (Right-tailed test).
- **Test Statistic:** $t = \frac{31.84 - 30.0}{5.48 / \sqrt{250}} = \frac{1.84}{0.3466} = 5.309$.
- **P-Value:** $p = 1.15 \times 10^{-7} < 0.05$.
- **Decision:** Reject $H_0$. Strong evidence that true delivery time exceeds SLA. Corrective driver allocation needed.

### Case 2: Website Landing Page Conversion Optimization
- **Context:** Growth team redesigns the onboarding checkout funnel. Historical benchmark conversion is $p_0 = 10.0\%$.
- **Data:** $n = 1,000$ unique visitor sessions, resulting in 122 conversions ($\hat{p} = 12.2\%$).
- **Hypotheses:** $H_0: p \le 0.10$ vs $H_1: p > 0.10$ (Right-tailed proportion test).
- **Standard Error:** $SE = \sqrt{\frac{0.10 \times 0.90}{1000}} = \sqrt{0.00009} = 0.009487$.
- **Test Statistic:** $z = \frac{0.122 - 0.10}{0.009487} = 2.319$.
- **P-Value:** $p = 0.0102 < 0.05$.
- **Decision:** Reject $H_0$. Statistically significant lift of $+2.2\%$ percentage points. Deploy Variant B globally.

### Case 3: High-Precision Pharmaceutical Manufacturing
- **Context:** An automated bottling line fills medical vials to target nominal weight $\mu = 500.0$ mg.
- **Data:** $n = 300$ sampled vials.
- **Sample Metrics:** $\bar{x} = 500.82$ mg, $s = 3.21$ mg.
- **Hypotheses:** $H_0: \mu = 500.0$ vs $H_1: \mu \ne 500.0$ (Two-tailed test).
- **Test Statistic:** $t = \frac{500.82 - 500.0}{3.21 / \sqrt{300}} = \frac{0.82}{0.1853} = 4.425$.
- **P-Value:** $p = 1.34 \times 10^{-5} < 0.05$.
- **Decision:** Reject $H_0$. Calibration has shifted higher by $+0.82$ mg. Immediate nozzle recalibration required.

### Case 4: E-Commerce Average Order Basket Value
- **Context:** Marketing department launches dynamic bundling to increase average transaction order value beyond $\$120.00$.
- **Data:** $n = 500$ transactions.
- **Sample Metrics:** $\bar{x} = \$125.42$, $s = \$28.51$.
- **Hypotheses:** $H_0: \mu \le 120.0$ vs $H_1: \mu > 120.0$ (Right-tailed test).
- **Test Statistic:** $t = \frac{125.42 - 120.0}{28.51 / \sqrt{500}} = \frac{5.42}{1.275} = 4.251$.
- **P-Value:** $p = 1.33 \times 10^{-5} < 0.05$.
- **Cohen's d:** $d = \frac{125.42 - 120.0}{28.51} = 0.190$ (Borderline small effect size).
- **Decision:** Reject $H_0$. Statistically significant lift, though practical effect size is modest.

---

## 🎯 4. Technical Interview Questions & Answers (30 Total)

#### Q1: What is the fundamental difference between the null hypothesis and alternative hypothesis?
**Answer:** The null hypothesis ($H_0$) represents the baseline claim of no effect, equality, or status quo. It always includes an equality sign ($=, \le, \ge$). The alternative hypothesis ($H_1$) is the research hypothesis claiming an active effect, difference, or directional change, characterized by strict inequalities ($
e, >, <$).

#### Q2: Why do statisticians say "fail to reject $H_0$" rather than "accept $H_0$"?
**Answer:** Because an experiment can only provide evidence *against* a hypothesis or fail to find sufficient evidence against it. Lack of sufficient evidence to disprove innocence does not mathematically prove innocence; it merely means sampling variation could plausible account for the observed data under $H_0$.

#### Q3: What is a Type I error, and what parameter controls its probability?
**Answer:** A Type I error is a false positive: rejecting the null hypothesis when it is actually true in reality. Its maximum probability is governed by the pre-specified significance level $\alpha$ (alpha).

#### Q4: What is a Type II error, and what is its relationship with statistical power?
**Answer:** A Type II error is a false negative: failing to reject the null hypothesis when it is actually false. Its probability is denoted by $\beta$ (beta). Statistical power is the complement, defined as $1 - \beta$, representing the probability of correctly rejecting a false null hypothesis.

#### Q5: Define a $p$-value precisely without using colloquialisms.
**Answer:** A $p$-value is the probability, conditioned on the null hypothesis being strictly true, of observing a test statistic at least as extreme as the test statistic calculated from the observed sample data: $P(T \ge t_{obs} \mid H_0)$.

#### Q6: Explain the common misconception: "A $p$-value of 0.03 means there is a 97% chance the alternative hypothesis is true."
**Answer:** This is the *fallacy of the transposed conditional*. The $p$-value calculates $P(\text{Data} \mid H_0)$, not $P(H_1 \mid \text{Data})$. Determining $P(H_1 \mid \text{Data})$ requires Bayesian inference incorporating prior probabilities ($P(H_0)$ and $P(H_1)$).

#### Q7: When should you use a 1-sample Z-test versus a 1-sample t-test?
**Answer:** Use a Z-test when the population standard deviation $\sigma$ is known and the sample size is large or data is normally distributed. Use a Student's t-test when the population standard deviation is unknown and estimated using the sample standard deviation $s$.

#### Q8: How do degrees of freedom affect the shape of the Student's t-distribution?
**Answer:** Degrees of freedom ($df = n - 1$) govern the thickness of the distribution's tails. For small $df$, the t-distribution has heavier tails than the standard normal distribution to account for uncertainty in estimating $\sigma$ with $s$. As $df \to \infty$, the t-distribution converges to the standard normal distribution.

#### Q9: What are the primary assumptions required for a valid 1-sample t-test?
**Answer:**
1. Continuous data measured on an interval or ratio scale.
2. Independent and identically distributed (i.i.d.) random sampling.
3. Normality of the underlying population or a sufficiently large sample size ($n \ge 30$) satisfying the Central Limit Theorem.
4. Absence of extreme outliers that distort the sample mean and variance.

#### Q10: How does a one-tailed test differ from a two-tailed test in terms of critical values and $p$-values?
**Answer:** For a significance level $\alpha = 0.05$:
- A two-tailed test splits $\alpha$ into two tails ($\alpha/2 = 0.025$ each), requiring critical values $\pm 1.96$ and doubling the one-tail $p$-value.
- A one-tailed test places the entire $\alpha = 0.05$ into one tail, requiring a critical value of $+1.645$ (or $-1.645$), resulting in higher power to detect effects in that specified direction.

#### Q11: What is the risk of using a one-tailed test?
**Answer:** If the true effect occurs in the opposite direction from what was hypothesized, a one-tailed test cannot reject $H_0$, no matter how massive the effect is in that opposite direction. Furthermore, deciding to use a one-tailed test *after* viewing the sample data inflates the Type I error rate.

#### Q12: Explain the duality between confidence intervals and two-tailed hypothesis tests.
**Answer:** A two-tailed hypothesis test at significance level $\alpha$ rejects $H_0: \mu = \mu_0$ if and only if $\mu_0$ lies outside the $(1 - \alpha)$ confidence interval. If $\mu_0$ falls within the CI, the test fails to reject $H_0$ at significance level $\alpha$.

#### Q13: What is Cohen's d, and why is it necessary alongside a $p$-value?
**Answer:** Cohen's $d = \frac{\bar{x} - \mu_0}{s}$ measures the standardized difference between the observed mean and null baseline in units of standard deviations. It measures practical effect size independent of sample size, whereas $p$-values depend heavily on sample size.

#### Q14: How can an experiment produce $p = 0.00001$ while being completely useless in practice?
**Answer:** When sample size $n$ is massive (e.g., $n = 500,000$), standard error becomes microscopic. A miniscule difference (e.g., page load time decreasing by 0.5 milliseconds) will yield a tiny $p$-value, but has zero business or experiential impact.

#### Q15: What four factors determine the statistical power ($1 - \beta$) of a hypothesis test?
**Answer:**
1. Effect size: Larger true effect sizes are easier to detect (higher power).
2. Sample size ($n$): Larger samples decrease standard error, increasing power.
3. Significance level ($\alpha$): A higher $\alpha$ (e.g. 0.10 vs 0.01) widens the rejection region, increasing power at the expense of more Type I errors.
4. Population variance ($\sigma^2$): Lower noise/variance increases power.

#### Q16: What is the rule of thumb for sample size in a 1-sample proportion Z-test?
**Answer:** The success-failure condition: $n p_0 \ge 10$ and $n (1 - p_0) \ge 10$, ensuring that the binomial sampling distribution is sufficiently symmetric and well-approximated by the Gaussian normal distribution.

#### Q17: What is continuity correction in proportion tests, and when is it applied?
**Answer:** Continuity correction subtracts or adds $0.5 / n$ to the sample proportion $\hat{p}$ to bridge the gap between a discrete binomial distribution and a continuous normal approximation. It is especially important for moderate sample sizes.

#### Q18: What is "p-hacking" (data dredging)?
**Answer:** The practice of repeatedly manipulating data collection, stopping rules, subgroup filtering, outlier exclusions, or testing dozens of hypotheses until an arbitrary threshold ($p < 0.05$) is achieved, resulting in spurious false positive findings.

#### Q19: What is the Bonferroni correction for multiple hypothesis testing?
**Answer:** If $m$ independent hypotheses are tested simultaneously, the family-wise error rate escalates ($1 - (1 - \alpha)^m$). The Bonferroni correction adjusts the significance threshold for each individual test to $\alpha^* = \alpha / m$.

#### Q20: Explain the difference between Fisher's view of $p$-values and Neyman-Pearson decision theory.
**Answer:**
- **Fisher:** Viewed the $p$-value as a continuous index of evidence against the null hypothesis in a single experiment, without formalizing an alternative hypothesis or Type II error.
- **Neyman-Pearson:** Formulated a rigorous operational decision framework with pre-specified $\alpha$ and $\beta$, choosing between two competing actions ($H_0$ vs $H_1$) to control long-run error rates.

#### Q21: What happens to the Type II error rate $\beta$ if you decrease the significance level $\alpha$ from 0.05 to 0.01?
**Answer:** As $\alpha$ decreases, the critical value moves further into the tails, making it harder to reject $H_0$. Consequently, the probability of failing to reject a false null hypothesis (Type II error $\beta$) increases, and statistical power ($1 - \beta$) decreases.

#### Q22: Can a t-test be used if the sample distribution is skewed?
**Answer:** If the sample size is sufficiently large ($n \ge 30-50$), the Central Limit Theorem ensures that the sampling distribution of the mean is approximately normal even if the underlying population is moderately skewed. For severe skewness or small samples ($n < 20$), a non-parametric test (e.g., Wilcoxon signed-rank test) or bootstrap test is preferred.

#### Q23: Why is standard error computed using $p_0$ rather than $\hat{p}$ in a 1-sample proportion hypothesis test?
**Answer:** Because the hypothesis test assumes $H_0$ is true as its starting premise. Under $H_0$, the true population proportion is known to be $p_0$, so the standard error under the null distribution must be $SE_0 = \sqrt{\frac{p_0(1 - p_0)}{n}}$.

#### Q24: What is the relation between a two-sample t-test and a one-sample t-test on paired differences?
**Answer:** A paired samples t-test computes differences $d_i = x_{1, i} - x_{2, i}$ for each matched pair and conducts a standard 1-sample t-test on the difference series against $\mu_d = 0$.

#### Q25: If an A/B test has an observed $p$-value of 0.06 with $\alpha = 0.05$, should you gather 50 more samples until it drops below 0.05?
**Answer:** No. This is "optional stopping", which inflates the Type I error rate well beyond the nominal 5%. Sample sizes must be fixed a priori via power analysis, or evaluated using sequential testing frameworks (e.g. Wald's SPRT).

#### Q26: What is the relationship between the test statistic $z$ and $t$ when $n = 10,000$?
**Answer:** They are practically indistinguishable. At $df = 9,999$, the critical t-value for $\alpha = 0.05$ (two-tailed) is $1.9602$, virtually identical to the Z-critical value of $1.95996$.

#### Q27: How does an extreme outlier in sample data affect the outcome of a t-test?
**Answer:** An outlier simultaneously distorts the sample mean $\bar{x}$ and dramatically inflates the sample standard deviation $s$. Because $s$ appears in the denominator ($s / \sqrt{n}$), the inflated variance often diminishes the t-statistic, falsely masking a true underlying effect.

#### Q28: What is the difference between a critical region and an acceptance region?
**Answer:** The critical (rejection) region is the set of test statistic values that leads to the rejection of $H_0$. The non-rejection region (sometimes informally called acceptance region) is the set of values for which sample data is compatible with $H_0$.

#### Q29: What is statistical power analysis used for before launching an experiment?
**Answer:** Sample size determination: calculating the minimum sample size $n$ required to achieve a desired power (typically $80\%$ or $90\%$) given a minimum detectable effect size (MDE), baseline variance, and significance level $\alpha$.

#### Q30: Why is pre-registration of hypothesis tests becoming standard practice in data science and research?
**Answer:** Pre-registration fixes the primary metric, sample size, hypothesis directionality, exclusion criteria, and statistical models before seeing the data, eliminating researcher degrees of freedom, confirmation bias, and p-hacking.

---

## 💡 5. 10 Deep Statistical Insights and Common Pitfalls

1. **Failure to Reject $\ne$ Proof of Null:** "Absence of evidence is not evidence of absence." A test with small sample size may have only 20% power; failing to reject $H_0$ simply means the test was underpowered to detect the effect.
2. **P-Value Threshold Fallacy:** There is no qualitative change in physical reality between $p = 0.049$ and $p = 0.051$. Treat $p$-values as continuous gradations of evidence rather than absolute cliffs.
3. **The Multiple Testing Trap:** Running 20 independent hypothesis tests at $\alpha = 0.05$ yields a $1 - (0.95)^{20} \approx 64.2\%$ probability of observing at least one spurious false positive by pure chance. Always adjust with Bonferroni or FDR corrections.
4. **Sample Size Asymmetry:** Huge sample sizes turn noise into statistical significance; tiny sample sizes bury huge effects in statistical noise. Always pair $p$-values with Cohen's d and confidence intervals.
5. **Post-Hoc Hypothesis Formulation:** Looking at data, noticing an unexpected bump, and formulating a directional one-tailed test on that bump invalidates standard probability theory.
6. **Normality Test Reliance:** Running Shapiro-Wilk or Kolmogorov-Smirnov before every t-test is often counterproductive: in small samples, normality tests lack power; in large samples, they flag harmless departures from normality that CLT renders irrelevant.
7. **Independence Violations:** If observations within a sample are correlated (e.g. repeated sessions from the same user, time-series autocorrelation), the true standard error is underestimated, resulting in wildly inflated Type I error rates.
8. **Asymmetry of Losses:** In business and medicine, Type I and Type II errors carry vastly different financial or human costs. Set $\alpha$ and $\beta$ according to the expected utility of the decision, not dogmatic 5% conventions.
9. **Confidence Intervals Tell the Whole Story:** If you report a 95% confidence interval alongside your point estimate, you provide the test decision, the estimate precision, and the range of plausible true parameter values in a single metric.
10. **The Base Rate Fallacy:** If only 1% of tested ideas are true innovations, even a test with $\alpha = 0.05$ and $80\%$ power will produce more false positive discoveries than true discoveries among all "statistically significant" results. Prior probability matters.

---
