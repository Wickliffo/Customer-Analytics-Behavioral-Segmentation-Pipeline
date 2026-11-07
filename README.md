
# Customer Behavioral Analytics & Segmentation Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary

Modern growth engineering requires moving beyond basic demographic segmentation toward data-driven, behavioral cohort analysis. This repository provides a complete analytical pipeline built in **Python** to segment an online learning platform's customer base ($N = 3,834$).

By synthesizing engagement metrics (`minutes_watched`), monetary valuation (`CLV`), acquisition channels (`Google`, `Facebook`, `LinkedIn`, `YouTube`, `Twitter`, etc.), and geographic regions, this project translates unsupervised machine learning models into high-yield ad spend recommendations.

---

## 📐 Problem Statement & Business Context

Marketing acquisition costs (CAC) continue to escalate across paid digital channels. Blanket advertising budgets yield sub-optimal return on ad spend (ROAS) when high-value customer profiles are not clearly delineated from casual users.

### Business Challenges:

1. **Inefficient Channel Allocation:** Budget spent uniformly across acquisition channels regardless of regional performance or Customer Lifetime Value (CLV).
2. **Engagement vs. Revenue Disconnect:** Failure to distinguish between high-engagement cohorts (power users) and high-monetary cohorts (high-ticket buyers).
3. **Unidentified Regional Yields:** Lack of visibility into which geographic markets deliver the highest conversion rates and long-term retention.

---

## 🛠️ Data Pipeline Architecture & Engineering Workflow

The pipeline transforms raw behavioral records into standardized feature matrices, applies distance-based clustering algorithms, and computes cluster profiles for executive decision-making.

```text
┌─────────────────────────────────────────────────────────┐
│              Raw Data Ingestion (CSV)                   │
│             (3,834 rows × 4 raw columns)                │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│             Data Cleaning & Imputation                  │
│       - minutes_watched missing values fillna(0)        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│             Feature Engineering (OHE)                   │
│       - One-Hot Encode acquisition channels (1-8)       │
│       - One-Hot Encode geographic regions (0-2)         │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│             Feature Standardization                     │
│           - StandardScaler (μ = 0, σ = 1)               │
└────────────────────────────┬────────────────────────────┘
                             │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ Hierarchical Clustering │     │   K-Means Clustering    │
│     (Ward's Linkage)    │     │     (Elbow Method)      │
└─────────────────────────┘     └────────────┬────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────┐
│        Cluster Profiling & Business Strategy            │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Exploratory Data Analysis & Descriptive Statistics

### Baseline Dataset Metrics ($N = 3,834$)

| Metric | `minutes_watched` | `CLV` ($) | `region` (Encoded) | `channel` (Encoded) |
| :--- | :--- | :--- | :--- | :--- |
| **Count** | 3,725 (109 imputed with 0) | 3,834 | 3,834 | 3,834 |
| **Mean** | 1,972.09 min | $118.38 | 1.17 | 3.87 |
| **Std Dev** | 7,401.69 min | $70.47 | 0.92 | 2.19 |
| **Min** | 0.00 min | $0.00 | 0 (`Region_0`) | 1 (`Google`) |
| **25%** | 159.00 min | $62.58 | 0 | 2 (`Facebook`) |
| **50% (Median)** | 512.00 min | $119.00 | 2 | 3 (`YouTube`) |
| **75%** | 1,521.00 min | $149.99 | 2 | 6 (`Instagram`) |
| **Max** | 288,508.00 min | $626.40 | 2 | 8 (`Other`) |

### Key Correlation Observations

* **`CLV` vs. `region`:** Shows a mild negative correlation ($-0.273$), indicating that geographic location exerts a measurable effect on overall monetary spend.
* **`minutes_watched` vs. `CLV`:** Weak positive linear correlation ($+0.114$), signaling that watch duration alone does not dictate financial valuation—necessitating non-linear cohort segmentation.

---

## 🤖 Modeling & Clustering Methodology

### 1. Feature Standardization

Due to large scale disparities between `minutes_watched` ($\max = 288,508$) and binary dummy variables ($0$ or $1$), feature normalization via `StandardScaler` is required to prevent scale distortion during Euclidean distance evaluation:

$$z = \frac{x - \mu}{\sigma}$$

### 2. Hierarchical Clustering

To inspect initial natural distance groupings, Ward’s minimum variance method was applied:

```python
from scipy.cluster.hierarchy import dendrogram, linkage

hier_clust = linkage(df_segmentation_scaled, method='ward')
```

### 3. K-Means Optimization (Elbow Method)

Evaluated Within-Cluster Sum of Squares (WCSS) across $k \in [1, 10]$ to isolate the optimal point of diminishing returns:

$$\text{WCSS} = \sum_{i=1}^{k} \sum_{x \in S_i} \Vert x - \mu_i \Vert^2$$

---

## 💡 Strategic Recommendations & Executive Impact

### Business Impact Matrix

| Cohort | Key Insight | Actionable Strategy |
| :--- | :--- | :--- |
| **Western Europe & Anglo-Saxon** | Highest average CLV across all regions. | Scale ad spend using a LinkedIn & Google/YouTube acquisition mix. |
| **Facebook Dedicated** | Avg >2,700 mins watched; moderate CLV. | Retain for community building & organic referrals; avoid high-CAC campaigns. |
| **Twitter Acquisition** | Low volume ($n=58$); poor conversion yield. | De-prioritize spend; reallocate test budget to emerging platforms (e.g., Threads). |
| **Rest of World (ROW)** | High organic reach potential. | Leverage LinkedIn certificate sharing to drive low-cost viral loops. |

---

## 💻 Repository Structure & Reproduction Guide

### Directory Layout

```text
├── data/
│   └── customer_segmentation_data.csv
├── notebooks/
│   └── customer-segment-analysis.ipynb
├── output/
│   ├── corr.png
│   ├── scatter.png
│   ├── dendrogram.png
│   └── elbow_method.png
├── .gitignore
├── README.md
└── requirements.txt
```

### Quickstart Execution

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Wickliffo/customer-segmentation-analytics.git
   cd customer-segmentation-analytics
   ```

2. **Set up a Python virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook:**

   ```bash
   jupyter notebook notebooks/customer-segment-analysis.ipynb
   ```

---

## 📜 Dependencies (`requirements.txt`)

```text
numpy>=1.20.0
pandas>=2.0.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 🤝 License & Author

Distributed under the MIT License. Designed and engineered by **Wickliffo**.
