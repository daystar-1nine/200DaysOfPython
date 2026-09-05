# 🚀 DAY 61 / 200 — Matplotlib Fundamentals: From Data Analysis to Data Visualization

---

## 🧭 Executive Summary & Core Philosophy

Over Days 51–60, we built a comprehensive foundation in Python data processing, numerical computation with **NumPy**, tabular data manipulation with **Pandas**, rigorous data cleaning, and exploratory data analysis (**EDA**). We computed descriptive statistics, grouped aggregations, correlation coefficients, rolling averages, and IQR outlier boundaries.

However, in professional enterprise environments, stakeholders rarely inspect raw tables or 50-row console matrices. **Data visualization** bridges computational analysis and executive decision-making:
> **"Data analysis tells us WHAT happened numerically. Data visualization reveals the hidden PATTERNS, TRAJECTORIES, and RELATIONSHIPS behind those numbers."**

On Day 61, we explore **Matplotlib**, the foundational data visualization library of the Python data science ecosystem.

---

## 🧠 1. The Architecture of Matplotlib

Matplotlib operates on a hierarchical object architecture:

```text
Figure (The Top-Level Canvas / Window)
└── Axes (The Coordinate System / Plot Area)
    ├── Title
    ├── X-Axis (Spines, Ticks, Tick Labels, Axis Label)
    ├── Y-Axis (Spines, Ticks, Tick Labels, Axis Label)
    ├── Data Artists (Lines, Bars, Patches, Collections, Text)
    ├── Legend
    └── Grid
```

### The Crucial Distinction: `Figure` vs `Axes` vs `Axis`
1. **`Figure`**: The top-level container that holds everything—canvases, subplots, colorbars, and global titles. Think of it as the physical sheet of paper or browser window.
2. **`Axes`**: The actual plotting area with a coordinate system. A `Figure` can contain one or multiple `Axes` objects (e.g., in a 2x2 subplot grid). **An `Axes` object is NOT the same as a geometric `Axis`.**
3. **`Axis`**: The numerical/categorical number lines (`ax.xaxis` and `ax.yaxis`) that manage ticks, tick markers, limits, and axis labels.

### Stateful (`pyplot`) vs Object-Oriented (OO) Interface
Matplotlib provides two distinct programming interfaces:
1. **Stateful Interface (`plt.plot`, `plt.title`, `plt.xlabel`)**:
   - Mirrors MATLAB syntax.
   - Keeps track of the "current" figure and axes implicitly in the background.
   - Convenient for quick one-line throwaway plots in an interactive terminal.
   - **Flaw**: Fragile, difficult to manage when working with multiple subplots, and prone to state leakage across modules.
2. **Object-Oriented (OO) Interface (`fig, ax = plt.subplots()`)**:
   - Explicitly creates and references `fig` and `ax` objects.
   - Every modification is an explicit method call on the target `ax` (`ax.plot`, `ax.set_title`, `ax.set_xlabel`).
   - **Gold Standard for Production**: Clean, testable, thread-safe, and effortlessly scalable to complex multi-panel dashboards.

```python
# Production-grade Object-Oriented Pattern
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3], [10, 25, 30], color="#1f77b4", linewidth=2.0, marker="o", label="Trajectory")
ax.set_title("Executive KPI Trajectory", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Quarter", fontsize=11)
ax.set_ylabel("Revenue (₹)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig("output/sample_plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)  # Mandatory: Reclaims memory allocated for figure
```

---

## 📊 2. Deep Dive: Core Chart Types & Use Cases

### 1. Line Charts (`ax.plot`)
- **Primary Use**: Visualizing continuous variables over an ordered sequence (time series, dates, sequential stages).
- **Key Parameters**: `color`, `linewidth`, `linestyle` (`"-"`, `"--"`, `":"`), `marker` (`"o"`, `"s"`, `"^"`), `markersize`, `label`, `alpha`.
- **Best Practice**: Limit lines on a single chart to 3–4 to prevent the "spaghetti chart" anti-pattern. Always provide a legend when plotting multiple series.

### 2. Vertical Bar Charts (`ax.bar`)
- **Primary Use**: Comparing discrete categories along a numerical metric.
- **Key Parameters**: `x`, `height`, `width`, `color`, `edgecolor`, `alpha`.
- **Best Practice**: Sort categories logically (e.g., descending by revenue) unless there is an inherent natural order (such as Age Brackets or Days of the Week).

### 3. Horizontal Bar Charts (`ax.barh`)
- **Primary Use**: Category comparisons where category labels are lengthy, preventing awkward 90-degree rotated x-axis text.
- **Key Parameters**: `y`, `width`, `height`, `color`, `edgecolor`.
- **Best Practice**: Place the highest-performing category at the top by reversing the categorical index or using `ax.invert_yaxis()`.

### 4. Scatter Plots (`ax.scatter`)
- **Primary Use**: Inspecting bivariate relationships, clustering, non-linear patterns, and potential correlations between two continuous variables.
- **Key Parameters**: `x`, `y`, `s` (size), `c` (color), `alpha` (transparency to diagnose point overplotting), `marker`, `edgecolors`.
- **Critical Caution**: Correlation \(\neq\) Causation. A strong linear pattern suggests co-movement, not causal dependency.

### 5. Histograms (`ax.hist`)
- **Primary Use**: Visualizing the distribution, spread, modality, and skewness of a single continuous variable.
- **Key Parameters**: `x`, `bins` (integer count or sequence of bin edges), `density` (normalize to probability density), `color`, `edgecolor`.
- **Binning Trade-off**: Too few bins over-smooth the data and conceal bi-modal behavior; too many bins create a noisy comb effect. Experiment with \(k = \sqrt{N}\) or Sturges' formula.

### 6. Pie Charts (`ax.pie`)
- **Primary Use**: Displaying part-to-whole proportions when the total sums strictly to 100%.
- **Key Parameters**: `x`, `labels`, `autopct="%1.1f%%"`, `startangle`, `colors`, `explode`.
- **Critical Caution**: Human visual perception is significantly worse at judging relative angles and areas than linear lengths. **Never use a pie chart for more than 5 categories.** When comparisons between slices are tight, a bar chart is objectively superior.

---

## 🛠️ 3. Production & Clean Visualization Principles

1. **Information-to-Ink Ratio**: Maximize the data-to-ink ratio (Edward Tufte). Eliminate gratuitous 3D effects, heavy borders, garish backgrounds, and decorative clutter.
2. **Explicit Labeling & Units**: Never assume the viewer knows the unit of measure. Always include currency symbols (e.g., `₹`), percentages (`%`), or physical units (`kg`, `hours`) in titles or axis labels.
3. **Non-Misleading Scales**: For bar charts, **always start the quantitative axis at zero**. Truncating the baseline exaggerates minor differences and creates visual deception.
4. **Memory Hygiene (`plt.close(fig)`)**: In batch scripts, automated report pipelines, and web servers, failing to call `plt.close(fig)` leaks memory because Matplotlib retains internal references to all created figures until the process terminates.
5. **Headless Execution (`matplotlib.use('Agg')`)**: When running in automated test runners (Pytest), CI/CD pipelines (GitHub Actions), Docker containers, or headless servers, ensure the non-GUI `Agg` (Anti-Grain Geometry) backend is configured before importing `pyplot`.

---

## 🎤 4. 25 Technical Interview Questions & In-Depth Answers

### Section 1: Matplotlib Architecture & Mechanics (Q1–Q10)

#### Q1: What is Matplotlib and why is it so widely adopted in Python?
**Answer**: Matplotlib is the primary low-level 2D/3D plotting and data visualization library in Python. Created by John D. Hunter in 2003, it was designed to emulate MATLAB's plotting capabilities within an open-source Python environment. It is universally adopted because it provides complete pixel-level control over every visual element (canvas, axes, spines, ticks, labels, annotations) and serves as the rendering foundation upon which higher-level libraries (like Seaborn and Pandas plotting) are built.

#### Q2: What is `matplotlib.pyplot`?
**Answer**: `matplotlib.pyplot` is a state-based procedural interface module within Matplotlib. It provides a MATLAB-like collection of command-style functions that make Matplotlib behave like a state machine, automatically creating figures, axes, and lines behind the scenes. In modular Python software, `pyplot` is used primarily as a factory to instantiate figures and axes via `plt.subplots()`.

#### Q3: What is a `Figure` in Matplotlib?
**Answer**: A `Figure` is the top-level container object representing the entire graphics window, web canvas, or saved image file. It manages global attributes such as physical dimensions (`figsize`), resolution (`dpi`), background color (`facecolor`), global title (`suptitle`), and holds references to all child `Axes`, colorbars, and legends.

#### Q4: What is an `Axes` in Matplotlib?
**Answer**: An `Axes` object (short for "Axes subplot") is the actual plotting surface within a `Figure` bounded by coordinate axes. It contains the data space where marks are rendered (lines, bars, scatter points) and encompasses the x-axis, y-axis, titles, tick marks, tick labels, and data limits. A single `Figure` can contain multiple `Axes` objects.

#### Q5: What is the difference between an `Axes` and an `Axis`?
**Answer**: 
- `Axes`: Represents the entire rectangular 2D/3D plotting region with coordinate systems, titles, legends, and plotted geometries.
- `Axis`: Refers specifically to an individual numerical/categorical dimension line within an `Axes` (i.e., `ax.xaxis` and `ax.yaxis`). The `Axis` handles tick mark positions, tick formatting, scale transformations (linear, log), and axis-specific labels.

#### Q6: Why is the Object-Oriented interface preferred over `plt.plot()` in production?
**Answer**: 
1. **Explicit Scoping**: When handling multiple plots or subplots, the OO approach (`fig, ax = plt.subplots()`) explicitly specifies which axis receives each chart element, preventing unintentional state mutation.
2. **Reusability & Modularization**: OO functions can accept an `ax` parameter directly (`def plot_trend(ax, data): ...`), enabling modular plot composition across services.
3. **Concurrency & Memory Safety**: Stateful `plt` calls rely on global internal state, leading to race conditions and cross-talk in multi-threaded or batch reporting environments.

#### Q7: How do you create and format a standard line chart in Matplotlib?
**Answer**:
```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_values, y_values, color="navy", linestyle="--", linewidth=1.8, marker="o", label="Metric")
ax.set_title("Line Chart Title")
ax.set_xlabel("Time Period")
ax.set_ylabel("Metric Value")
ax.grid(True, alpha=0.3)
ax.legend()
```

#### Q8: How do you create a vertical bar chart and annotate values on top of each bar?
**Answer**:
```python
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, values, color="teal")
ax.bar_label(bars, fmt="₹%.0f", padding=3, fontsize=9)
ax.set_ylim(0, max(values) * 1.15)  # Leave headroom for labels
```

#### Q9: How do you create a scatter plot and handle point overplotting?
**Answer**:
```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, s=50, color="crimson", alpha=0.6, edgecolors="none")
```
The `alpha` parameter introduces transparency (0.0 to 1.0). Overlapping data points create darker clusters, allowing viewers to detect high-density regions that would otherwise be obscured.

#### Q10: How do you create a histogram and choose bin counts appropriately?
**Answer**:
```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=15, color="steelblue", edgecolor="black", alpha=0.8)
```
Bin counts can be set via an integer (`bins=15`), explicit bin boundaries (`bins=[0, 10, 25, 50, 100]`), or automated statistical heuristics like Sturges' rule (`bins="sturges"`) or Freedman-Diaconis (`bins="fd"`).

---

### Section 2: Visualization Principles & Formatting (Q11–Q20)

#### Q11: When should you use a line chart versus a bar chart?
**Answer**:
- **Line Chart**: Used when the independent variable is continuous or ordered (time series, continuous dates, distance, temperature trends). The connecting line implies a continuous progression between successive points.
- **Bar Chart**: Used for discrete, unordered, or distinct categorical variables (Departments, Regions, SKUs, Customer Segments) where connecting lines would misleadingly imply continuous interpolation between unrelated entities.

#### Q12: Why and when should you use a horizontal bar chart (`barh`) instead of a vertical bar chart?
**Answer**: A horizontal bar chart is preferred when:
1. Category names are lengthy (e.g., product titles, book titles, department descriptions). Vertical charts force awkward 45° or 90° rotations that impair reading speed.
2. Ranking items (e.g., Top 10 SKUs), as vertical scanning mimics natural document reading from top to bottom.

#### Q13: What is the fundamental difference between a Histogram and a Bar Chart?
**Answer**:
- **Bar Chart**: Compares discrete categorical groups. The bars have distinct gaps between them to indicate independent categories, and the x-axis has no continuous scale.
- **Histogram**: Visualizes the continuous probability or frequency distribution of a single numerical variable. The numerical range is partitioned into adjacent, continuous intervals (bins) with no spaces between bars (unless a bin frequency is zero).

#### Q14: When is a scatter plot the most appropriate visualization?
**Answer**: A scatter plot is best when analyzing bivariate relationships between two continuous numerical variables to detect:
1. Direction of relationship (positive, negative, or zero correlation).
2. Form of relationship (linear, quadratic, exponential, asymptotic).
3. Strength of association (tight clustering along a line vs widely dispersed clouds).
4. Outliers and bivariate anomalies (points departing drastically from the general cluster).

#### Q15: Why are pie charts widely discouraged by professional data analysts, and when (if ever) are they acceptable?
**Answer**: 
- **Why Discouraged**: Humans perceive linear lengths and 2D positions far more accurately than angles or 2D slice areas. When slices have similar percentages (e.g., 26% vs 24%), visual differentiation is nearly impossible without data labels.
- **When Acceptable**: Acceptable only when displaying 2 to 3 categories with drastically unequal, easily distinguishable shares (e.g., Active 85% vs Inactive 15%) strictly representing 100% of a whole.

#### Q16: Why are descriptive titles, axis labels, and units of measure mandatory in charts?
**Answer**: A chart must be **self-contained and unambiguous**. Without explicit axis titles and units (e.g., `Revenue (in Millions ₹)` vs `Revenue (USD)`), viewers cannot accurately interpret magnitude. Descriptive titles should answer *What*, *Where*, and *When*, often summarizing the key insight rather than merely naming the variables.

#### Q17: When is a legend necessary, and when does it become visual clutter?
**Answer**:
- **Necessary**: When multiple data series share the same visual space and encode information via color, line style, or marker shape (e.g., Actual Revenue vs Target vs Moving Average).
- **Visual Clutter**: When there is only a single data series, or when categories are already labeled directly on the x-axis or adjacent to the bars. Redundant legends violate Tufte's data-ink maximization rule.

#### Q18: How do you save a high-resolution figure to disk in Matplotlib?
**Answer**:
```python
fig.savefig("output/chart.png", dpi=300, bbox_inches="tight", transparent=False)
```
- `dpi=300`: Ensures print/publication resolution (300 dots per inch).
- `bbox_inches="tight"`: Recalculates canvas boundaries to ensure external labels, titles, and legends are not cropped out.

#### Q19: What does `figsize` control, and what units does it accept?
**Answer**: `figsize=(width, height)` controls the physical canvas dimensions in **inches**. When multiplied by `dpi` (dots per inch), it determines the exact pixel resolution of the exported bitmap:
$$\text{Pixel Width} = \text{figsize}[0] \times \text{dpi}$$
$$\text{Pixel Height} = \text{figsize}[1] \times \text{dpi}$$
For example, `figsize=(10, 6)` at `dpi=300` generates a \(3000 \times 1800\) pixel image.

#### Q20: What does `dpi` mean, and why is it important for reporting?
**Answer**: `dpi` stands for **Dots Per Inch** (pixels per inch). Screens typically display at 72–96 DPI, whereas print media and high-DPI displays require 300+ DPI. Setting an appropriate DPI prevents blurriness, pixelation, and artifacts when embedding figures in PDF reports, slides, or web dashboards.

---

### Section 3: Data Analytics & Production Practices (Q21–Q25)

#### Q21: How would you visualize monthly revenue and moving trends together?
**Answer**: Using a multi-line chart where:
1. Solid line with low opacity or distinct marker shows actual volatile monthly revenue (`ax.plot(months, revenue, marker="o", alpha=0.6, label="Actual")`).
2. Thicker, contrasting solid line shows the 3-month rolling average (`ax.plot(months, rolling_rev, color="red", linewidth=2.5, label="3-Month Trend")`).
3. Shaded fill between lines or target bands highlights variance and momentum.

#### Q22: How would you visualize category performance when there are 15+ categories?
**Answer**: Use a **horizontal bar chart (`barh`)** sorted descending by performance metric:
1. Long category labels remain horizontal and easily legible.
2. The top-performing and bottom-performing categories are immediately identifiable at the extremes.
3. Color-code top performers (e.g., top 3 in blue, others in muted gray) to direct audience attention.

#### Q23: How would you visualize the distribution of employee salaries?
**Answer**: Combine a **histogram** with a **Kernel Density Estimate (KDE)** or boxplot:
1. The histogram reveals bin frequencies and skewness (typically right-skewed for compensation).
2. Vertical dashed lines mark the **Mean** and **Median** to highlight positive skewness ($Mean > Median$).
3. Annotate 25th, 50th, and 75th percentiles to communicate compensation quartiles.

#### Q24: How would you investigate a suspected relationship between Marketing Spend and Revenue?
**Answer**:
1. Plot a bivariate **scatter plot** of Spend ($X$) vs Revenue ($Y$).
2. Calculate and display the **Pearson correlation coefficient** ($r$) on the chart.
3. Overlay an OLS linear trendline (`np.polyfit`) to assess linearity and heteroscedasticity (expanding variance).
4. Inspect points that diverge significantly from the line as potential high-ROI or inefficient campaigns.

#### Q25: Why must visualization strictly come AFTER data cleaning and validation?
**Answer**: 
1. **Outlier Distortion**: Uncleaned extreme values or erroneous dummy inputs (e.g., `-999` or `999999`) compress the visible scale, flattening legitimate data into an unreadable line.
2. **Category Splintering**: Unstandardized strings (`"West"`, `"west"`, `"West "` ) generate duplicate, fragmented bars, misrepresenting total volume.
3. **Null Breakages**: Unhandled `NaN` values cause silent gaps in line charts or crash histogram binning algorithms.
4. **Misleading Decisions**: Visualizing corrupted data communicates false patterns with visual authority, leading to flawed business strategies.

---

## 📝 5. Day 61 Revision Test & Solution

### Objective
Given the following synthetic dataset:
```python
import pandas as pd
df = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Sales": [20000, 24000, 27000, 25000, 32000, 38000],
    "Profit": [4000, 5000, 6000, 4500, 7000, 9000]
})
```
Build:
1. Sales line chart.
2. Profit line chart.
3. Combined Sales vs Profit chart.
4. Title and axis labels.
5. Legend.
6. Grid where useful.
7. Save chart to disk.
8. Provide two actionable business insights.

### Implementation
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Sales": [20000, 24000, 27000, 25000, 32000, 38000],
    "Profit": [4000, 5000, 6000, 4500, 7000, 9000]
})

fig, ax = plt.subplots(figsize=(10, 6))

# Plot Sales and Profit series
ax.plot(df["Month"], df["Sales"], marker="o", linewidth=2.2, color="#1f77b4", label="Sales (₹)")
ax.plot(df["Month"], df["Profit"], marker="s", linewidth=2.2, color="#2ca02c", linestyle="--", label="Profit (₹)")

# Formatting
ax.set_title("H1 Performance: Monthly Sales vs Profit", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Month", fontsize=11, fontweight="bold")
ax.set_ylabel("Financial Amount (₹)", fontsize=11, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left", frameon=True)

# Annotate endpoints
ax.annotate(f"₹{df['Sales'].iloc[-1]:,}", (df["Month"].iloc[-1], df["Sales"].iloc[-1]), textcoords="offset points", xytext=(-10, 10), ha="center", fontweight="bold")
ax.annotate(f"₹{df['Profit'].iloc[-1]:,}", (df["Month"].iloc[-1], df["Profit"].iloc[-1]), textcoords="offset points", xytext=(-10, 10), ha="center", fontweight="bold")

fig.tight_layout()
fig.savefig("output/revision_test_sales_profit.png", dpi=300, bbox_inches="tight")
plt.close(fig)
```

### Actionable Business Insights
1. **Parallel Growth Trajectory**: Both Sales (+90.0% from Jan to Jun) and Profit (+125.0% from Jan to Jun) demonstrate robust upward expansion across H1, with profit expanding faster than top-line revenue, indicating positive operational leverage.
2. **April Contraction & Rebound**: In April, both sales experienced a temporary -7.4% dip and profit declined -25.0%. However, May and June rebounded aggressively to all-time highs, proving that the April slump was a transient seasonal correction rather than a structural downturn.
