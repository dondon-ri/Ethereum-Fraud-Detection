import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

# ==============================================================================
# 1. Load and Split Data
# ==============================================================================
df = pd.read_csv('cleaned_ethereum_data.csv')
X = df.drop('FLAG', axis=1)
y = df['FLAG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==============================================================================
# 2. Setup Scaler
# ==============================================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ==============================================================================
# 3. Define and Train Initial 4 Models
# ==============================================================================

#scale_weight for XGboost
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
estimated_contamination = num_pos / len(y_train)
xgb_scale_weight = num_neg / num_pos

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss',scale_pos_weight=xgb_scale_weight, random_state=42),
    "Isolation Forest": IsolationForest(contamination=0.1, random_state=42)
}

model_performance = []
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

for i, (name, model) in enumerate(models.items()):
    if name == "Isolation Forest":
        model.fit(X_train)
        y_pred = [1 if p == -1 else 0 for p in model.predict(X_test)]
        if_scores = model.decision_function(X_test_scaled)
    elif name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Performance Mapping
    report = classification_report(y_test, y_pred, output_dict=True)
    model_performance.append({
        "Model": name,
        "F1-Score": report['1']['f1-score'],
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "Accuracy": round(report['accuracy'], 4)
    })

    # Confusion Matrix Rendering
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f"{name}\nConfusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

# Adding dynamic presentation context notes at the baseline
fig.text(0.5, 0.02,
         "Note: 0 = Normal Transaction | 1 = Fraud Case\n"
         "Top-Left: Correctly flagged Normal | Top-Right: False Alarms (Innocent flagged) \n"
         "Bottom-Left: Missed Fraud (Thieves escaped) | Bottom-Right: Correctly caught Fraud",
         ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()

# ==============================================================================
# 4. Feature Space Allocation (Top 5, Top 10, Top 15)
# ==============================================================================
all_cols = X_train.columns.tolist()

def get_col(keyword):
    match = [c for c in all_cols if keyword.lower() in c.lower()]
    return match[0] if match else None

try:
    # Top 5 Highly Correlated Features
    top_5_raw = [
        get_col('Time Diff between first and last'),
        get_col('Avg min between received tnx'),
        get_col('total transactions (including tnx'),
        get_col('Received Tnx'),
        get_col('Sent tnx')
    ]

    # Top 10 Features
    top_10_raw = top_5_raw + [
        get_col('avg val sent'),
        get_col('ERC20 uniq rec token name'),
        get_col('ERC20 uniq rec contract addr'),
        get_col('Unique Sent To Addresses'),
        get_col('Total ERC20 tnxs')        
    ]

    # Top 15 Features
    top_15_raw = top_10_raw + [
        get_col('Unique Received From Addresses'),
        get_col('Avg min between sent tnx'),
        get_col('ERC20 uniq rec addr'),
        get_col('ERC20 uniq sent token name'),
        get_col('ERC20 uniq sent addr')
    ]

    # FIXED: Clean and parse exact lists to clear structural bugs from prior version
    top_5 = list(dict.fromkeys([c for c in top_5_raw if c]))
    top_10 = list(dict.fromkeys([c for c in top_10_raw if c]))
    top_15 = list(dict.fromkeys([c for c in top_15_raw if c]))

except Exception as e:
    print(f"Mapping error: {e}")
    top_5, top_10, top_15 = all_cols[:5], all_cols[:10], all_cols[:15]

# Update targeted feature sets dictionary configuration
feature_sets = {
    "Top 5 Features": top_5,
    "Top 10 Features": top_10,
    "Top 15 Features": top_15,
    "All Features": all_cols
}

# ==============================================================================
# 5. Run Targeted Multi-Subset Feature Experimentations
# ==============================================================================
exp_results = []
for label, cols in feature_sets.items():
    # 1. Train XGBoost Variant
    xgb_exp = XGBClassifier(eval_metric='logloss', scale_pos_weight=xgb_scale_weight,random_state=42)
    xgb_exp.fit(X_train[cols], y_train)
    y_pred_xgb = xgb_exp.predict(X_test[cols])

    # 2. Train Random Forest Variant
    rf_exp = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_exp.fit(X_train[cols], y_train)
    y_pred_rf = rf_exp.predict(X_test[cols])

    # Metrics Logging Configuration
    exp_results.append({
        "Set": label,
        "Model": "XGBoost",
        "F1-Score": round(f1_score(y_test, y_pred_xgb), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred_xgb), 4)
    })

    exp_results.append({
        "Set": label,
        "Model": "Random Forest",
        "F1-Score": round(f1_score(y_test, y_pred_rf), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred_rf), 4)
    })

df_exp = pd.DataFrame(exp_results)

# ==============================================================================
# 6. Performance Reports Outputs
# ==============================================================================
print("\n----4-models performance comparison----")
print(pd.DataFrame(model_performance))

print("\n----XGB & RF performance based on features 5,10,15,all----")
print(df_exp.to_string(index=False))

# ==============================================================================
# 7. Anomaly Distribution Diagnostics (Isolation Forest Visual)
# ==============================================================================
plt.figure(figsize=(12, 6))
sns.histplot(if_scores[y_test == 0], color='blue', label='Actual Normal', kde=True, element="step")
sns.histplot(if_scores[y_test == 1], color='red', label='Actual Fraud', kde=True, element="step")

plt.title("Isolation Forest: Identifying Unseen Fraud Patterns", fontsize=15)
plt.xlabel("Anomaly Score (Lower/Left = More 'Weird' or Suspicious)", fontsize=12)
plt.ylabel("Number of Transactions")
plt.legend()
plt.savefig('isolation_forest_analysis.png', dpi=300)
plt.show()

# ==============================================================================
# 8. Train & Export Final Top-10 Production Model to Streamlit
# ==============================================================================
print("\nTraining final Top-10 Production Variant for Streamlit Integration...")
final_production_model = XGBClassifier(eval_metric='logloss', random_state=42)
final_production_model.fit(X_train[top_10], y_train)

# Save the explicitly isolated Top-10 model asset and scaler
joblib.dump(final_production_model, 'ethereum_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Success: Top-10 Model and Scaler assets saved for Streamlit deployment!")