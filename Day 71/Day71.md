# 🚀 DAY 71 / 200 — Two-Sample Tests & A/B Testing

**Progress:** 71 / 200 → **35.5% complete**  
**Remaining:** **129 days**  
**Milestone:** Transition from single-parameter inference to comparative experiments and online A/B testing!

---

## 🧭 1. Executive Summary & Why Two-Sample Testing Matters

Yesterday you mastered hypothesis testing for a single population against a fixed theoretical benchmark (e.g. testing whether delivery times exceed 30 minutes).

In modern Data Science, product development, and business analytics, questions rarely involve fixed external constants. Instead, you are tasked with answering comparative, causal questions:
- *"Does the new checkout redesign convert better than our current layout?"*
- *"Does the updated recommendation algorithm increase Average Revenue Per User (ARPU) compared to the baseline model?"*
- *"Did the employee training program significantly improve customer resolution times before versus after the intervention?"*
- *"Is there a significant difference in churn rates between iOS and Android users?"*

**Two-Sample Tests and A/B Testing** form the scientific gold standard for causal inference. They allow data scientists to isolate treatment effects, control for confounding variables via randomization, and determine whether an observed lift represents **genuine product improvement** or merely **stochastic sampling variation**.

---

## 🏛️ 2. Core Foundations & Theoretical Framework

### 2.1 The Two Fundamental Experimental Designs: Independent vs Paired

Before calculating any test statistic, you must identify the **data-generating process** that produced the two groups:

```text
                    EXPERIMENTAL DESIGN ARCHITECTURE
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
     INDEPENDENT SAMPLES                         PAIRED SAMPLES
(Between-Subjects Experiment)             (Within-Subjects Experiment)
               │                                       │
• Two distinct cohorts of subjects        • Same subjects measured twice
• Observations in A are independent       • Observations in A and B are
  of observations in B                      statistically correlated
• Examples:                               • Examples:
  - Control users vs Treatment users        - Before training vs After training
  - Male vs Female shoppers                 - Pre-op vs Post-op blood pressure
  - Group A (US) vs Group B (UK)            - Same user testing Design A & B
               │                                       │
        Welch's t-Test                           Paired t-Test
   Two-Proportion Z-Test                   (One-sample t on differences)
```

#### The Mathematical Power of Pairing (Variance Reduction)
When you measure the same individual before and after a treatment, each subject acts as their own control. The variance of the difference between two variables is:

$$\text{Var}(X_1 - X_2) = \text{Var}(X_1) + \text{Var}(X_2) - 2\text{Cov}(X_1, X_2)$$

In independent samples, the covariance is zero: $\text{Cov}(X_1, X_2) = 0$, so the variances add up.  
In paired samples with a positive correlation (e.g., fast employees before training remain relatively fast after training), $\text{Cov}(X_1, X_2) > 0$. The subtraction of the covariance term **dramatically shrinks the standard error**, yielding vastly higher statistical power for the same sample size!

---

### 2.2 Comparing Two Independent Means

When comparing numerical metrics (such as basket totals, session times, or page load speeds) between two independent cohorts, two versions of the two-sample t-test exist:

#### 1. Student's Two-Sample t-Test (Equal Variance Assumed)
Assumes that both populations have identical variance ($\sigma_1^2 = \sigma_2^2$). It computes a pooled sample variance:

$$s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}$$

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2 \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}, \quad df = n_1 + n_2 - 2$$

#### 2. Welch's Two-Sample t-Test (Unequal Variance — Modern Industry Standard)
Does **not** assume equal population variances. The standard error is computed directly:

$$SE_{\bar{x}_1 - \bar{x}_2} = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

The degrees of freedom are calculated using the **Welch-Satterthwaite equation**:

$$df \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$

> [!TIP]
> **Modern Best Practice:** Always default to **Welch's t-test** (`equal_var=False` in SciPy). If the variances happen to be equal, Welch's test loses virtually zero power compared to Student's test; but if the variances differ and sample sizes are unequal, Student's test severely inflates Type I error rates.

---

### 2.3 Paired Two-Sample t-Test

When observations are paired (e.g. $n$ subjects with Before and After measurements):
1. Compute the intra-individual difference for each subject:
   $$d_i = \text{After}_i - \text{Before}_i$$
2. Compute the sample mean difference and standard deviation of differences:
   $$\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i, \quad s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2}$$
3. Conduct a one-sample t-test testing $H_0: \mu_d = 0$ against $H_1: \mu_d \ne 0$:
   $$t = \frac{\bar{d} - 0}{\frac{s_d}{\sqrt{n}}} = \frac{\bar{d}\sqrt{n}}{s_d}, \quad df = n - 1$$

---

### 2.4 Comparing Two Proportions in A/B Testing

In product experimentation, the most common metric is a **conversion rate** (binomial proportion):
- Control: $n_A$ visitors, $x_A$ conversions $\implies \hat{p}_A = \frac{x_A}{n_A}$
- Treatment: $n_B$ visitors, $x_B$ conversions $\implies \hat{p}_B = \frac{x_B}{n_B}$

#### Hypotheses
$$H_0: p_A = p_B \quad \iff \quad H_0: p_B - p_A = 0$$
$$H_1: p_A \ne p_B \quad (\text{two-sided}) \quad \text{or} \quad H_1: p_B > p_A \quad (\text{right-sided})$$

#### The Pooled Proportion Test Statistic
Under the null hypothesis $H_0$, both groups share the exact same underlying conversion rate. Therefore, we pool the successes:

$$\hat{p} = \frac{x_A + x_B}{n_A + n_B}$$

$$\text{SE}_{\text{pool}} = \sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}$$

$$z = \frac{\hat{p}_B - \hat{p}_A}{\text{SE}_{\text{pool}}}$$

---

### 2.5 Confidence Intervals for Differences

Statistical estimation is essential alongside hypothesis testing. We report confidence intervals to communicate the plausible magnitude of the effect:

#### Difference in Means (Independent Groups)
$$(\bar{x}_B - \bar{x}_A) \pm t_{\alpha/2, df} \cdot \sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}$$

#### Difference in Proportions (Unpooled for Estimation)
For confidence intervals, we do **not** assume $p_A = p_B$, so we use unpooled standard error:

$$\text{SE}_{\text{unpool}} = \sqrt{\frac{\hat{p}_A(1 - \hat{p}_A)}{n_A} + \frac{\hat{p}_B(1 - \hat{p}_B)}{n_B}}$$

$$(\hat{p}_B - \hat{p}_A) \pm z_{\alpha/2} \cdot \text{SE}_{\text{unpool}}$$

---

### 2.6 Absolute Lift vs Relative Lift

In business communication, mixing up absolute and relative lift is one of the most frequent sources of stakeholder confusion:

| Metric | Formula | Example ($5.2\% \to 5.8\%$) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Absolute Lift** | $\Delta p = \hat{p}_B - \hat{p}_A$ | $5.8\% - 5.2\% = +0.6\%$ | Conversion rate improved by **0.6 percentage points**. |
| **Relative Lift** | $\text{RelLift} = \frac{\hat{p}_B - \hat{p}_A}{\hat{p}_A} \times 100\%$ | $\frac{0.6\%}{5.2\%} \times 100\% \approx +11.54\%$ | Conversion rate improved by **11.54% relative to baseline**. |

---

### 2.7 Standardized Effect Size: Cohen's d

When sample sizes are massive ($n = 100,000$), microscopic differences yield tiny $p$-values ($p < 0.0001$). Standardized effect size allows us to evaluate **practical significance** independent of sample size:

#### Independent Samples:
$$d = \frac{\bar{x}_B - \bar{x}_A}{s_{pooled}}, \quad \text{where } s_{pooled} = \sqrt{\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}$$

#### Paired Samples:
$$d = \frac{\bar{d}}{s_d}$$

| Absolute Cohen's $d$ | Effect Magnitude | Real-World Meaning |
| :--- | :--- | :--- |
| $|d| < 0.20$ | Negligible | Trivial effect; rarely economically viable on its own |
| $0.20 \le |d| < 0.50$ | Small | Noticeable only through rigorous instrumentation |
| $0.50 \le |d| < 0.80$ | Medium | Substantial effect noticeable in standard operations |
| $|d| \ge 0.80$ | Large | Massive effect obvious to casual observation |

---

## 🔬 3. Professional A/B Testing Architecture

A/B testing is not merely running a statistical formula; it is a rigorous engineering and decision framework:

```text
                  END-TO-END A/B TESTING WORKFLOW
                                 │
                   1. Hypothesis & MDE Planning
                                 │
                      2. Random Assignment
                    (50% Control / 50% Treatment)
                                 │
                  3. Sanity Check: SRM Audit
                  (Chi-square test on user counts)
                                 │
                   4. Primary Metric Evaluation
                 (Two-Proportion Z-Test on Conversion)
                                 │
                   5. Guardrail Metric Audits
             (Bounce Rate, Refund Rate, Latency)
                                 │
                    6. Business Impact Sizing
               (Incremental Orders & Revenue)
                                 │
                   7. Executive Recommendation
             (Launch, Iterate, or Abandon Variant)
```

### 3.1 The Role of Guardrail Metrics
Optimizing solely for a primary metric is perilous. Consider these scenarios:
1. **Aggressive upsell popup:** Conversion rate increases by $+10\%$, but bounce rate surges by $+35\%$ and session duration plummets.
2. **Deceptive checkout pre-checks:** Revenue rises by $+15\%$, but customer refund requests and chargebacks jump by $+80\%$.

A robust experimentation engine requires **Guardrail Constraints**:
$$\text{Launch Recommendation} = (p \le \alpha) \land (\text{RelLift} \ge \text{MDE}) \land (\text{Guardrails Met})$$

---

## 🚨 4. Common A/B Testing Pitfalls & How to Prevent Them

1. **The "Peeking" Problem (Optional Stopping):**
   - Continuously monitoring $p$-values daily and stopping the test the instant $p < 0.05$ inflates the actual false positive rate from $5\%$ to over $30\%$!
   - *Remedy:* Pre-determine sample size via power analysis and evaluate only once $N$ is reached, or use sequential testing procedures (e.g. Always Valid $P$-values).
2. **Sample Ratio Mismatch (SRM):**
   - Intended split is 50/50, but observed traffic is 48/52. This signals broken redirects, bot traffic, or tracking drop-off.
   - *Remedy:* Run a Chi-square goodness-of-fit test on group counts before analyzing metrics.
3. **The Multiple Comparisons Trap:**
   - Testing 20 different segments or metrics at $\alpha = 0.05$ produces a $1 - (0.95)^{20} \approx 64\%$ chance of at least one false positive.
   - *Remedy:* Designate a single Primary Metric upfront; apply Bonferroni or False Discovery Rate (FDR) corrections to secondary metrics.
4. **Novelty & Primacy Effects:**
   - Existing users initially click a new button simply because it looks different (novelty), but behavior reverts after 10 days.
   - *Remedy:* Run experiments across full business cycles (at least 1–2 full weeks) and track cohort behavior over time.
5. **Simpson's Paradox:**
   - Aggregated data shows Treatment winning, but broken down by device (Mobile vs Desktop), Control wins in both categories because mobile traffic shifted.
   - *Remedy:* Stratified randomization and subgroup interaction checks.

---

## 🎯 5. Technical Interview Questions & Answers (30 Total)

#### Q1: What is the fundamental difference between an independent samples t-test and a paired t-test?
**Answer:** An independent samples t-test compares two separate, unrelated cohorts (e.g., Control vs Treatment). A paired t-test compares two dependent measurements taken from the exact same subjects or matched pairs (e.g., Before vs After). Pairing removes inter-subject variability by analyzing intra-individual differences ($d_i$).

#### Q2: When and why should you use Welch's t-test over Student's standard independent t-test?
**Answer:** Welch's t-test does not assume equal population variances and uses the Welch-Satterthwaite equation to adjust degrees of freedom. In modern practice, Welch's t-test is the recommended default because it maintains nominal Type I error rates when variances and sample sizes differ, while losing negligible power when variances happen to be equal.

#### Q3: Why is standard error pooled when conducting a two-proportion hypothesis test, but unpooled when constructing a confidence interval?
**Answer:** Under the null hypothesis $H_0: p_1 = p_2$, both populations are assumed to have the exact same proportion $p$, so pooling successes provides the best estimate of the common standard error under $H_0$. For a confidence interval, we are estimating the difference without assuming $H_0$ is true, so each group's individual sample proportion $\hat{p}_1$ and $\hat{p}_2$ must be used.

#### Q4: Explain the difference between absolute lift and relative lift with an example.
**Answer:** If Control conversion is $5.0\%$ and Treatment is $6.0\%$:
- Absolute lift is $\hat{p}_T - \hat{p}_C = 6.0\% - 5.0\% = +1.0$ percentage points.
- Relative lift is $\frac{6.0\% - 5.0\%}{5.0\%} \times 100\% = +20.0\%$ increase relative to baseline.

#### Q5: What is the "peeking problem" in A/B testing?
**Answer:** Checking $p$-values repeatedly throughout an active experiment and stopping as soon as $p < 0.05$. Because sampling error fluctuates randomly, the test is guaranteed to cross the $0.05$ threshold by chance at some point during the run, severely inflating the Type I error rate (often up to $20\%-40\%$).

#### Q6: What is a Sample Ratio Mismatch (SRM) and why is it dangerous?
**Answer:** SRM occurs when the observed ratio of users assigned to Control vs Treatment deviates significantly from the planned randomization design (e.g., 50/50 intended, but 48/52 observed). It indicates an underlying bias or technical defect (e.g., variant crashes on certain browsers, bot filtering discrepancies), rendering all subsequent statistical conclusions invalid.

#### Q7: How do you mathematically test for Sample Ratio Mismatch (SRM)?
**Answer:** Using a Chi-Square Goodness-of-Fit test:
$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$
For a 50/50 split of $N$ users, $E_A = E_B = N/2$. If the resulting $p$-value is $< 0.001$, an SRM is detected.

#### Q8: What is Cohen's d for independent samples, and what are the standard benchmark thresholds?
**Answer:** Cohen's $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$ measures the distance between two group means in units of pooled standard deviation. Standard benchmarks: $|d| < 0.20$ (Negligible), $0.20 \le |d| < 0.50$ (Small), $0.50 \le |d| < 0.80$ (Medium), $|d| \ge 0.80$ (Large).

#### Q9: What is a guardrail metric in experimentation?
**Answer:** A metric tracked alongside the primary objective to ensure that gains in one area do not compromise the overall product health or customer experience (e.g., page load latency, unsubscribe rates, bounce rates, customer service ticket volume, refund rates).

#### Q10: Why can a statistically significant result fail to be practically significant in business?
**Answer:** With very large sample sizes (e.g. millions of impressions), the standard error becomes tiny, making even an infinitesimal difference (e.g. $+0.01\%$ conversion lift) statistically significant ($p < 0.01$). However, the financial benefit may be far too small to justify the engineering, maintenance, or infrastructure costs of deploying the change.

#### Q11: How does sample size affect statistical power in a two-sample test?
**Answer:** Increasing sample size decreases the standard error of the difference ($SE \propto 1/\sqrt{n}$), narrowing the distribution of the test statistic under $H_1$ and increasing the probability of rejecting a false null hypothesis ($1 - \beta$).

#### Q12: What is Simpson's Paradox in A/B testing?
**Answer:** A statistical phenomenon where a treatment appears to outperform control in aggregate, but underperforms control within every subgroup (e.g. Mobile vs Desktop). This occurs when the subgroup composition is unbalanced across variants.

#### Q13: What is the unit of randomization in an e-commerce A/B test?
**Answer:** The entity randomly assigned to variants—most commonly User ID (for logged-in users) or Cookie ID / Device ID (for anonymous visitors). Randomizing at the session level causes user pollution, where the same user sees different variants across visits.

#### Q14: What is the multiple comparisons problem, and how is it mitigated?
**Answer:** Testing multiple variants or metrics simultaneously inflates the family-wise error rate ($FWER = 1 - (1 - \alpha)^m$). It is mitigated using Bonferroni correction ($\alpha^* = \alpha / m$), Holm-Bonferroni, or False Discovery Rate (Benjamini-Hochberg) methods.

#### Q15: Why is pre-determining experiment duration necessary even if target sample size is reached in 2 days?
**Answer:** To capture full business cycle seasonality (weekday vs weekend patterns, pay-day effects). Experiments should run for full 7-day cycles (typically 1–2 weeks) to avoid day-of-week bias.

#### Q16: What assumptions are required for Welch's two-sample t-test?
**Answer:** Continuous data, independent observations between and within groups, and approximately normal distributions in both populations (or large sample sizes $n_1, n_2 \ge 30$ by the Central Limit Theorem).

#### Q17: What assumptions are required for a two-sample proportion Z-test?
**Answer:** Independent random sampling and the success-failure condition in both groups: $n_A p_A \ge 10$, $n_A (1-p_A) \ge 10$, $n_B p_B \ge 10$, and $n_B (1-p_B) \ge 10$.

#### Q18: What is the difference between a one-tailed and two-tailed A/B test?
**Answer:** A two-tailed test evaluates whether the treatment is significantly different (better OR worse) than control ($\alpha$ split between both tails). A one-tailed test evaluates only whether the treatment is strictly better. Industry standard overwhelmingly favors two-tailed tests to detect regressions and avoid post-hoc bias.

#### Q19: What is the difference between an A/A test and an A/B test?
**Answer:** In an A/A test, both groups receive the identical existing control experience. It is used to validate the experimentation infrastructure, verify that the false positive rate matches $\alpha = 0.05$, and confirm the absence of SRM or tracking bugs.

#### Q20: How do you calculate the 95% confidence interval for the difference between two proportions?
**Answer:**
$$(\hat{p}_B - \hat{p}_A) \pm 1.96 \times \sqrt{\frac{\hat{p}_A(1 - \hat{p}_A)}{n_A} + \frac{\hat{p}_B(1 - \hat{p}_B)}{n_B}}$$

#### Q21: What is a novelty effect in product experimentation?
**Answer:** A temporary surge in user engagement caused solely by the visual novelty of a feature rather than its underlying utility. Novelty effects typically dissipate after 1–2 weeks as user familiarity returns.

#### Q22: Can a paired t-test be used if sample sizes in Before and After groups are unequal?
**Answer:** No. A paired test requires matched pairs; every subject must have both a Before and an After measurement. Missing pairs must either be imputed or dropped.

#### Q23: What is the Minimum Detectable Effect (MDE)?
**Answer:** The smallest true effect size that an experiment is powered to detect with a given probability (power, typically 80%) at a chosen significance level ($\alpha = 0.05$). Smaller MDEs require exponentially larger sample sizes ($N \propto 1 / MDE^2$).

#### Q24: What is the relationship between the two-sample t-test and simple linear regression with a binary dummy variable?
**Answer:** They are mathematically identical. Regressing outcome $Y$ on dummy variable $X \in \{0, 1\}$ produces an OLS slope coefficient $\hat{\beta}_1 = \bar{Y}_1 - \bar{Y}_0$, and its t-statistic matches the Student's two-sample pooled t-test exactly.

#### Q25: Why is random assignment the foundation of causal inference?
**Answer:** Random assignment ensures that both observed and unobserved confounding variables (user age, device type, purchasing power) are balanced across Control and Treatment groups in expectation, isolating the variant as the sole causal driver of differences.

#### Q26: What is a cannibalization effect in an A/B test?
**Answer:** When an increase in the primary metric for one product or feature comes at the direct expense of another feature (e.g. promoting Product X boosts its sales but reduces sales of higher-margin Product Y).

#### Q27: How does high variance in revenue affect sample size requirements in A/B testing?
**Answer:** Revenue metrics are notoriously skewed with heavy tails (few whales spending large amounts). High variance vastly inflates the standard error, requiring substantially larger sample sizes compared to binary conversion rates.

#### Q28: How does user pollution happen in experimentation?
**Answer:** When a single user interacts with the product on multiple devices or browsers and is randomly assigned to Control on one and Treatment on another, contaminating clean behavioral measurements.

#### Q29: What is Average Revenue Per User (ARPU) and how is it compared between groups?
**Answer:** $\text{ARPU} = \frac{\text{Total Revenue}}{\text{Total Users}}$ (including non-converting users who spend \$0). Because ARPU is continuous and highly skewed, it is tested using Welch's t-test, Mann-Whitney U, or non-parametric bootstrapping.

#### Q30: What decision should be made if an A/B test has $p = 0.03$, $+5\%$ relative lift, but refund rate increases from $2\%$ to $8\%$?
**Answer:** Do **not** launch. Even though the conversion increase is statistically significant, the severe violation of the refund rate guardrail metric signals customer dissatisfaction and increased operational costs, resulting in a net negative business outcome.

---

## 💡 6. 10 Deep Statistical Insights and Common Pitfalls

1. **The P-Value Asymmetry:** A small $p$-value tells you the data is unlikely under $H_0$; it does **not** prove the treatment caused a massive business win. Always evaluate Cohen's d, relative lift, and confidence intervals.
2. **Never Ignore the Guardrails:** A feature that doubles signups by hiding pricing details will inevitably destroy retention and inflate churn. Experimentation requires holistic guardrail governance.
3. **The Danger of Aggregated AOV:** Calculating Average Order Value (AOV) only among converted users introduces selection bias, because the composition of converting users may have changed between variants. Focus on ARPU across all randomized users.
4. **Early Stopping is a Silent Killer:** Even when a test looks "massively significant" on Day 2, weekday vs weekend composition shifts and novelty effects often erase the lead by Day 14.
5. **Sample Ratio Mismatch is a Hard Blocker:** Never proceed with analysis if SRM test $p$-value $< 0.001$. Fix the assignment bug and rerun the experiment.
6. **The Power Law of Sample Size:** Halving the Minimum Detectable Effect (MDE) requires a $4\times$ increase in sample size ($N \propto 1 / MDE^2$).
7. **Pairing Cancels Noise:** Whenever intra-subject repeated measurements are feasible, choose a paired design. The elimination of between-subject variability yields massive statistical power.
8. **Welch's t-Test Should Be Your Default:** In modern empirical data science, assuming equal variances is an unnecessary gamble. Welch's test protects against variance heterogeneity with virtually no penalty.
9. **Conversion vs Revenue Trade-Off:** High-converting low-ticket items can increase conversion rates while lowering total revenue. Track both conversion and financial yield.
10. **A/A Testing Validates Your Engine:** Run continuous A/A tests. If your experimentation platform rejects $H_0$ more than $5\%$ of the time at $\alpha = 0.05$, your infrastructure has systematic bias.

---
