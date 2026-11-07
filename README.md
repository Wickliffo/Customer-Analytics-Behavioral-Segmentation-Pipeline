# Customer-Analytics-Behavioral-Segmentation-Pipeline
📌 Executive Summary

Modern growth engineering requires moving beyond basic demographic segmentation toward data-driven, behavioral cohort analysis. This repository provides a complete, production-grade analytical pipeline built in Python to segment an online learning platform's customer base ($N = 3,834$).

By synthesizing engagement metrics (minutes_watched), monetary valuation (CLV), acquisition channels (Google, Facebook, LinkedIn, YouTube, Twitter, etc.), and geographic regions, this project translates unsupervised machine learning models into high-yield ad spend recommendations.

📐 Problem Statement & Business Context

Marketing acquisition costs (CAC) continue to escalate across paid digital channels. Blanket advertising budgets yield sub-optimal return on ad spend (ROAS) when high-value customer profiles are not clearly delineated from casual users.

Business Challenges:

Inefficient Channel Allocation: Budget spent uniformly across acquisition channels regardless of regional performance or Customer Lifetime Value (CLV).

Engagement vs. Revenue Disconnect: Failure to distinguish between high-engagement cohorts (power users) and high-monetary cohorts (high-ticket buyers).

Unidentified Regional Yields: Lack of visibility into which geographic markets deliver the highest conversion rates and long-term retention.

🛠️ Data Pipeline Architecture & Engineering Workflow

The pipeline transforms raw behavioral records into standardized feature matrices, applies distance-based clustering algorithms, and computes cluster profiles for executive decision-making.

┌────────────────────────────────┐
│   Raw Data Ingestion (CSV)     │
│   (3,834 rows x 4 raw cols)    │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│    Data Cleaning & Imputation  │
│  - minutes_watched fillna(0)   │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│   Feature Engineering (OHE)    │
│  - One-Hot Encode: channel (1-8)│
│  - One-Hot Encode: region (0-2) │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│    Feature Standardization     │
│  - StandardScaler (μ=0, σ=1)   │
└───────────────┬────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐ ┌───────────────┐
│ Hierarchical  │ │    K-Means    │
│ Clustering    │ │  Clustering   │
│ (Ward Link)   │ │  (Elbow Method│
└───────────────┘ └───────┬───────┘
                        │
                        ▼
┌────────────────────────────────┐
│   Profile Extraction & Business│
│   Optimization Insights        │
└────────────────────────────────┘


📊 Exploratory Data Analysis & Descriptive Statistics

Baseline Dataset Metrics ($N = 3,834$)

Metric

minutes_watched

CLV ($)

Region (Encoded)

Channel (Encoded)

Count

3,725 (109 missing imputed with 0)

3,834

3,834

3,834

Mean

1,972.09 min

$118.38

1.17

3.87

Std Dev

7,401.69 min

$70.47

0.92

2.19

Min

0.00 min

$0.00

0 (Region_0)

1 (Google)

25%

159.00 min

$62.58

0

2 (Facebook)

50% (Median)

512.00 min

$119.00

2

3 (YouTube)

75%

1,521.00 min

$149.99

2

6 (Instagram)

Max

288,508.00 min

$626.40

2

8 (Other)

Key Correlation Observations

CLV vs. Region: Shows a mild negative correlation ($-0.273$), indicating that geographic location exerts a measurable effect on monetary spend.

Minutes Watched vs. CLV: Weak positive linear correlation ($+0.114$), signaling that watch duration alone does not dictate financial valuation—necessitating non-linear cohort segmentation.

🤖 Modeling & Clustering Methodology

1. Feature Standardization

Due to large scale disparities between minutes_watched ($\max = 288,508$) and binary dummy variables ($0$ or $1$), feature normalization via StandardScaler is required to prevent scale distortion during Euclidean distance evaluation:

$$z = \frac{x - \mu}{\sigma}$$

2. Hierarchical Clustering

To inspect initial natural distance groupings, Ward’s minimum variance method was applied:

from scipy.cluster.hierarchy import dendrogram, linkage

hier_clust = linkage(df_segmentation_scaled, method='ward')


3. K-Means Optimization (Elbow Method)

Evaluated Within-Cluster Sum of Squares (WCSS) across $k \in [1, 10]$ to isolate the optimal point of diminishing returns:

$$\text{WCSS} = \sum_{i=1}^{k} \sum_{x \in S_i} \Vert{}x - \mu_i\Vert{}^2$$

💡 Strategic Recommendations & Executive Impact

                          BUSINESS IMPACT SUMMARY
 ┌───────────────────────┬──────────────────────┬─────────────────────────┐
 │ Cohort                │ Key Insight          │ Actionable Strategy     │
 ├───────────────────────┼──────────────────────┼─────────────────────────┤
 │ Western Europe &      │ Highest average CLV  │ Double down on LinkedIn │
 │ Anglo-Saxon           │ across all regions.  │ & Google/YouTube mix.   │
 ├───────────────────────┼──────────────────────┼─────────────────────────┤
 │ Facebook Dedicated    │ Avg >2,700 mins;     │ Retain for community;   │
 │                       │ moderate CLV.        │ avoid high CAC campaigns│
 ├───────────────────────┼──────────────────────┼─────────────────────────┤
 │ Twitter Acquisition   │ Low volume (n=58);   │ Reallocate spend to     │
 │                       │ poor conversion yield│ emerging networks.      │
 └───────────────────────┴──────────────────────┴─────────────────────────┘


1. Reallocate Capital to High-CLV Markets (Western Europe & Anglo-Saxon)

Finding: Customers acquired from Western Europe (Region 1) and Anglo-Saxon territories yield the highest CLV across all clusters.

Strategy: Scale ad spend in these regions. Primary acquisition channels in these areas are LinkedIn, Google, and YouTube.

2. Differentiate Engagement from Monetization (Facebook Cohort)

Finding: The Facebook-acquired cluster represents $\sim 8\%$ of the customer base. While their financial contribution is moderate, their engagement is extreme (averaging over 2,700 minutes watched).

Strategy: Leverage this segment for product feedback, testimonial generation, and organic referral loops rather than aggressive upselling.

3. De-prioritize Low-Performing Paid Channels (Twitter)

Finding: The Twitter cohort contains only 58 observations ($\approx 1.5\%$ of total users) with poor monetary conversion.

Strategy: Reduce Twitter acquisition spend to a baseline maintenance level. Pivot experimental ad budgets toward emerging networks (e.g., Threads, Meta Ads).

4. Organic Growth Acceleration (Rest of the World)

Finding: LinkedIn is the single most effective conversion channel for international regions outside North America and Western Europe, heavily driven by user certificate sharing.

Strategy: Optimize Product-Led Growth (PLG) mechanics—such as auto-generated LinkedIn certificate badges—to bolster organic user acquisition.

💻 Repository Structure & Reproduction Guide

Directory Layout

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


Quickstart Execution

Clone the repository:

git clone https://github.com/Wickliffo/customer-segmentation-analytics.git
cd customer-segmentation-analytics


Set up a Python virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook notebooks/customer-segment-analysis.ipynb


📜 Dependencies (requirements.txt)

numpy>=1.20.0
pandas>=2.0.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0


🤝 License & Author

Distributed under the MIT License. Designed and engineered by Wickliffo.
