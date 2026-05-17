import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import numpy as np

# 1. Load and Split Data
df = pd.read_csv('cleaned_ethereum_data.csv')
X = df.drop('FLAG', axis=1)
y = df['FLAG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Setup Scaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 3. Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "Isolation Forest": IsolationForest(contamination=0.1, random_state=42)
}

model_performance = []

# 4. Train and Evaluate Models
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

    report = classification_report(y_test, y_pred, output_dict=True)
    model_performance.append({
        "Model": name,
        "F1-Score": report['1']['f1-score'],
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "Accuracy": round(report['accuracy'], 4)
    })

print("\n---- 4-models performance comparison ----")
print(pd.DataFrame(model_performance))

# ==============================================================================
# 5. FIXED: PLOT ACTUAL FEATURE IMPORTANCES FOR XGBOOST (ALL FEATURES)
# ==============================================================================
print("\nGenerating XGBoost Feature Importance Visualization...")
trained_xgb_model = models["XGBoost"]

xgb_importances = trained_xgb_model.feature_importances_
xgb_indices = np.argsort(xgb_importances)[::-1] # Sort from highest to lowest

plt.figure(figsize=(12, 6))
plt.title("XGBoost - Actual Feature Importances (All Features)", fontsize=14, fontweight='bold')
plt.bar(range(X_train.shape[1]), xgb_importances[xgb_indices], color='#50C878', align="center")
# Using X_train.columns because XGBoost was trained on raw unscaled X_train in your loop
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in xgb_indices], rotation=90)
plt.ylabel("Importance Weight")
plt.tight_layout()
plt.show()

# ==============================================================================
# 6. ADDED: PLOT ACTUAL FEATURE IMPORTANCES FOR RANDOM FOREST (ALL FEATURES)
# ==============================================================================
print("\nGenerating Random Forest Feature Importance Visualization...")
trained_rf_model = models["Random Forest"]

rf_importances = trained_rf_model.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1] # Sort from highest to lowest

plt.figure(figsize=(12, 6))
plt.title("Random Forest - Actual Feature Importances (All Features)", fontsize=14, fontweight='bold')
plt.bar(range(X_train.shape[1]), rf_importances[rf_indices], color='#3498db', align="center")
# Using X_train.columns because Random Forest was also trained on raw unscaled X_train in your loop
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in rf_indices], rotation=90)
plt.ylabel("Importance Weight (Gini)")
plt.tight_layout()
plt.show()

# ==============================================================================
# EXPERIMENTAL ABLATION STUDY WITH FUZZY COLUMNS MATCHING (dropping key features)
# ==============================================================================
print("\n--- Running Experimental Ablation Study ---")

# 1. key features to drop
keywords_to_drop = [
    'Sent tnx',
    'ERC 20 max val rec',
    'ERC20 most sent token type',
    'total ether received',
    'max value received'
]

# 2. Automatically find the real column names in the actual dataset
features_to_drop = []
all_actual_columns = X_train.columns.tolist()

for keyword in keywords_to_drop:
    # Standardize the keyword: lowercase and remove spaces
    clean_keyword = keyword.lower().replace(" ", "")

    # Search actual dataset columns for a match
    match = [
        col for col in all_actual_columns
        if clean_keyword in col.lower().replace(" ", "")
    ]

    if match:
        features_to_drop.append(match[0])  # Snag the exact column name
        print(f"Found and mapped: '{keyword}' -> Actual column: '{match[0]}'")
    else:
        print(f"Warning: Could not find any column matching '{keyword}'")

# 3. if matches were found
if len(features_to_drop) > 0:
    print(f"\nDropping {len(features_to_drop)} features from the dataset...")
    X_train_broken = X_train.drop(columns=features_to_drop)
    X_test_broken = X_test.drop(columns=features_to_drop)

    # Train the experimental model
    xgb_experimental = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_experimental.fit(X_train_broken, y_train)

    # Calculate performance drop
    y_pred_broken = xgb_experimental.predict(X_test_broken)
    broken_f1 = f1_score(y_test, y_pred_broken)

    print("=" * 50)
    print(f"Original XGBoost F1-Score: 0.9504")
    print(f"Experimental F1-Score WITHOUT these features: {broken_f1:.4f}")
    print(f"Performance Drop: {0.9504 - broken_f1:.4f}")
    print("=" * 50)
else:
    print("\nExperiment aborted: Zero matching features were located.")
    print("=" * 50)