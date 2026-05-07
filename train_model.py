import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

#LOAD AND CLEAN DATA
df = pd.read_csv('transaction_dataset.csv')
to_drop = ['Unnamed: 0', 'Index', 'Address', ' ERC20 most sent token type', ' ERC20_most_rec_token_type']
df_clean = df.drop(columns=to_drop).fillna(0)

# Remove columns with only one value
single_val_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
df_clean = df_clean.drop(columns=single_val_cols)

X = df_clean.drop('FLAG', axis=1)
y = df_clean['FLAG']

# SPLIT AND SCALE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaler for Logistic Regression (Math-based models need this)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DEFINE MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# TRAIN AND COLLECT DATA
results_list = []
cms = []  # To store confusion matrices for the combined plot

for name, model in models.items():
    print(f"\n--- Training {name} ---")

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    #Print the detailed report for model
    print(f"Detailed Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

    #Save Metrics for the final comparison table
    report = classification_report(y_test, y_pred, output_dict=True)
    results_list.append({
        "Model": name,
        "Precision": round(report['1']['precision'], 4),
        "Recall": round(report['1']['recall'], 4),
        "F1-Score": round(report['1']['f1-score'], 4),
        "Accuracy": round(report['accuracy'], 4)
    })

    # C. Save Confusion Matrix for the combined plot
    cms.append((name, confusion_matrix(y_test, y_pred)))

#PRINT AND SAVE COMPARISON RESULTS
comparison_df = pd.DataFrame(results_list)
print("\n" + "="*30)
print("FINAL MODEL COMPARISON TABLE")
print("="*30)
print(comparison_df)
comparison_df.to_csv('model_comparison_results.csv', index=False)

#SHOW COMBINED CONFUSION MATRICES (Side-by-Side)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, cm) in enumerate(cms):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'Confusion Matrix:\n{name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

    #Adding explanatory notes at the bottom ---
    fig.text(0.5, 0.02,
             "Note: 0 = Normal Transaction | 1 = Fraud Case\n"
             "Top-Left: Correctly flagged Normal | Top-Right: False Alarms (Innocent flagged) \n"
             "Bottom-Left: Missed Fraud (Thieves escaped) | Bottom-Right: Correctly caught Fraud",
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

plt.subplots_adjust(bottom=0.2)
plt.show()

#PERFORMANCE COMPARISON CHART
comparison_df.set_index('Model')[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison (Class 1: Fraud)')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

#FEATURE IMPORTANCE (for Random Forest)
importances = pd.Series(models['Random Forest'].feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Most Important Features (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()