# Dashboard Fridays

A compilation of dashboards I build every friday of the week.

Each dashboard uses a different real-world dataset to build progressively advanced Power BI skills; from basic layout to DAX, multipage reports, drill through, bookmarks, and executive storytelling.

## Why
Power Bi is core skill for data roles. This repo demostrate my skill in real analysis with real datasets.

## Dashboards

### Week One - Fraud Analytics Overview

![Dashboard Overview](assets/fraud_analytics_overview.PNG)

#### Dataset
**Name:** Credit Card Fraud Detection
**Source:** [Kaggle - Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

#### What I built
A single page fraud overview dashboard showing the scale, distribution, amd timing of fraudulent credit transactions.

### Week Two - Model Threshold Analysis

![Dashboard Overview](assets/model_threshold_analysis.PNG)

### Dataset
**Name:** Credit Card Fraud Detection
**Source:** [Kaggle - Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

#### What I built
An interactive threshold analysis dashboard that lets stakeholders drag a
slicer and instantly see how changing the model decision boundary affects
precision, recall, F1 score, and the full confusion matrix. Designed to
demonstrate why "accuracy" is meaningless for imbalanced datasets and how
the precision-recall trade-off drives real business decisions.

#### Disclaimer
There is no native model probability in the dataset — had to engineer
ModelScore from PCA components (V1, V2), which gives a plausible but
imperfect simulation
