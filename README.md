# Ethereum Fraud Detection ML project

## Dataset

Ethereum Fraud Detection Dataset from **Kaggle**
https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

## Model Performance Summary

We evaluated three different algorithms to find the best balance between security and user experience.

| Model                   | Precision | Recall | F1-Score | Accuracy |
| :---------------------- | :-------- | :----- | :------- | :------- |
| **Gradient Boosting**   | 0.9688    | 0.8532 | 0.9073   | 0.9614   |
| **Random Forest**       | 0.9761    | 0.8417 | 0.9039   | 0.9604   |
| **Logistic Regression** | 0.3488    | 0.8601 | 0.4964   | 0.6135   |

## Key Fraud Indicators

Our model identified the following "Top 3" patterns that most commonly indicate fraudulent behavior:

1. **Account Lifespan (Time Diff):** Malicious "burner" accounts typically have a very short lifespan between their first and last transactions.
2. **Total Ether Received:** Fraudulent wallets often show massive, sudden spikes in incoming volume that deviate from normal user patterns.
3. **Average Receipt Value:** High-value, singular transfers are more characteristic of theft payouts than the varied, smaller transactions of regular users.
