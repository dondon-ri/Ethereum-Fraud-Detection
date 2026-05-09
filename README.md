# Ethereum Fraud Detection ML project

## Dataset

Ethereum Fraud Detection Dataset from **Kaggle**
https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

## Model Performance Summary

We evaluated three different algorithms to find the best balance between security and user experience.

| Model                   | Precision | Recall | F1-Score | Accuracy |
| :---------------------- | :-------- | :----- | :------- | :------- |
| **Logistic Regression** | 0.3488    | 0.8601 | 0.4964   | 0.6135   |
| **Random Forest**       | 0.9761    | 0.8417 | 0.9039   | 0.9604   |
| **XGBoost**             | 0.9847    | 0.8876 | 0.9336   | 0.9721   |
| **Isolation Forest**    | 0.0733    | 0.0321 | 0.0447   | 0.6958   |

## Testing Top2 models with selected features sets

Based on the performance comparison of 4 models, we choose the Top2 performaner that is XGBoost and Random Forest. After that we train each model with selected features category like Top3,5,15 and all features to see will the model perform with just a set of selected features and to test if other features are just data nosie. We choose the features from correlation heapmap.
