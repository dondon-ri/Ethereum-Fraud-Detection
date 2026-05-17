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

df = pd.read_csv('cleaned_ethereum_data.csv')

X = df.drop('FLAG', axis=1)
y = df['FLAG']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# 4 models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ),

    "Isolation Forest": IsolationForest(
        contamination=0.1,
        random_state=42
    )
}

# Storage for results
model_performance = []
cms = []

# ==============================
# CONFUSION MATRICES
# ==============================

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

    # Detailed Reports
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    model_performance.append({
        "Model": name,
        "F1-Score": report['1']['f1-score'],
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "Accuracy": round(report['accuracy'], 4)
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[i]
    )

    axes[i].set_title(f"{name}\nConfusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

# Explanation Notes
fig.text(
    0.5,
    0.02,
    "Note: 0 = Normal Transaction | 1 = Fraud Case\n"
    "Top-Left: Correctly flagged Normal | Top-Right: False Alarms (Innocent flagged)\n"
    "Bottom-Left: Missed Fraud (Thieves escaped) | Bottom-Right: Correctly caught Fraud",
    ha='center',
    fontsize=12,
    bbox=dict(
        facecolor='white',
        alpha=0.5,
        edgecolor='black'
    )
)

plt.subplots_adjust(bottom=0.2)

# SAVE CONFUSION MATRIX
fig.savefig(
    "confusion_matrices.png",
    dpi=300,
    bbox_inches='tight'
)

plt.tight_layout()
plt.show()

# ==============================
# FEATURE TESTING
# ==============================

all_cols = X_train.columns.tolist()

# Helper to find exact column names
def get_col(keyword):

    match = [
        c for c in all_cols
        if keyword.lower() in c.lower()
    ]

    return match[0] if match else None

try:

    # Top 3
    top_3 = [
        get_col('Time Diff between first and last'),
        get_col('Avg min between received tnx'),
        get_col('total transactions (including tnx')
    ]

    # Top 5
    top_5 = top_3 + [
        get_col('Received Tnx'),
        get_col('Sent tnx')
    ]

    # Top 15
    top_15 = top_5 + [
        get_col('avg val sent'),
        get_col('ERC20 uniq rec token name'),
        get_col('ERC20 uniq rec contract addr'),
        get_col('Unique Sent To Addresses'),
        get_col('Total ERC20 tnxs'),
        get_col('Unique Received From Addresses'),
        get_col('Avg min between sent tnx'),
        get_col('ERC20 uniq rec addr'),
        get_col('ERC20 uniq sent token name'),
        get_col('ERC20 uniq sent addr')
    ]

    # Remove None + duplicates
    top_3 = list(dict.fromkeys([c for c in top_3 if c]))
    top_5 = list(dict.fromkeys([c for c in top_5 if c]))
    top_15 = list(dict.fromkeys([c for c in top_15 if c]))

except Exception as e:

    print(f"Mapping error: {e}")

    top_3 = all_cols[:3]
    top_5 = all_cols[:5]
    top_15 = all_cols[:15]

# Feature sets
feature_sets = {
    "Top 3 Features": top_3,
    "Top 5 Features": top_5,
    "Top 15 Features": top_15,
    "All Features": all_cols
}

# ==============================
# EXPERIMENTS
# ==============================

exp_results = []

for label, cols in feature_sets.items():

    # XGBoost
    xgb_exp = XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )

    xgb_exp.fit(X_train[cols], y_train)

    y_pred_xgb = xgb_exp.predict(X_test[cols])

    # Random Forest
    rf_exp = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )

    rf_exp.fit(X_train[cols], y_train)

    y_pred_rf = rf_exp.predict(X_test[cols])

    # Store results
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

# Results dataframe
df_exp = pd.DataFrame(exp_results)

# ==============================
# FINAL SUMMARY
# ==============================

print("\n----4-models performance comparison----")
print(pd.DataFrame(model_performance))

print("\n----XGB & RF performance based on features 3,5,15,all----")
print(df_exp.to_string(index=False))

# ==============================
# ISOLATION FOREST ANALYSIS
# ==============================

plt.figure(figsize=(12, 6))

sns.histplot(
    if_scores[y_test == 0],
    color='blue',
    label='Actual Normal',
    kde=True,
    element="step"
)

sns.histplot(
    if_scores[y_test == 1],
    color='red',
    label='Actual Fraud',
    kde=True,
    element="step"
)

plt.title(
    "Isolation Forest: Identifying Unseen Fraud Patterns",
    fontsize=15
)

plt.xlabel(
    "Anomaly Score (Lower/Left = More 'Weird' or Suspicious)",
    fontsize=12
)

plt.ylabel("Number of Transactions")

plt.legend()

# SAVE ISOLATION FOREST FIGURE
plt.savefig(
    'isolation_forest_analysis.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()