# Bank Transaction Clustering and Classification Project

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised%20%26%20Unsupervised-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange)

## Overview
This project performs customer segmentation through clustering analysis and builds classification models to predict transaction clusters. The workflow includes:

1. **Unsupervised Learning**: K-Means clustering to group transactions
2. **Supervised Learning**: Classification models to predict clusters

## Dataset
- **Source**: [Kaggle - Bank Transaction Dataset](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)
- **Records**: 2,512 transactions
- **Features**:
  - Numerical: TransactionAmount, AccountBalance, TransactionDuration
  - Categorical: AgeGroup, CustomerOccupation, TransactionType

## Clustering Analysis
### Methodology
1. Data preprocessing:
   - Age grouping (18-40, 41-60, 61-80)
   - Label encoding for categorical features
   - Standard scaling for numerical features

2. Optimal cluster determination:
   - Elbow Method (k=3)
   - Silhouette Score: 0.4613

3. K-Means implementation with PCA visualization

### Cluster Characteristics
| Cluster | Age Group | Occupation | Avg Balance | Transaction Pattern |
|---------|-----------|------------|-------------|---------------------|
| 0       | 41-60     | Doctor     | $9,142      | High-value          |
| 1       | 18-40     | Student    | $1,641      | Frequent, low-value |
| 2       | 61-80     | Retired    | $4,076      | Moderate activity   |

## Classification Models
### Performance Comparison
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 99.60%   | 0.99      | 0.99   | 0.99     |
| Decision Tree       | 99.20%   | 0.99      | 0.99   | 0.99     |
| Random Forest       | 99.01%   | 0.99      | 0.99   | 0.99     |
| KNN                 | 69.98%   | 0.70      | 0.71   | 0.70     |

### Best Model
**Logistic Regression** achieved the highest accuracy (99.6%) with parameters:
```python
{'C': 10, 'max_iter': 100, 'solver': 'liblinear'}
