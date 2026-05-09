import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('transaction_dataset.csv')

# Drop non-numeric and unhelpful columns
to_drop = ['Unnamed: 0', 'Index', 'Address', ' ERC20 most sent token type', ' ERC20_most_rec_token_type']
df_clean = df.drop(columns=to_drop).fillna(0)

# Drop columns with zero variance
single_val_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
df_clean = df_clean.drop(columns=single_val_cols)

# Calculate the correlation matrix
corr_matrix = df_clean.corr()

# Force 'FLAG' to be the first row and first column for easy reading
cols = ['FLAG'] + [c for c in corr_matrix.columns if c != 'FLAG']
corr_matrix = corr_matrix.loc[cols, cols]

plt.figure(figsize=(24, 18))  # Large size to handle many features

sns.heatmap(corr_matrix, 
            annot=True,      
            cmap='coolwarm', 
            linewidths=0.5,
            fmt=".2f",
            annot_kws={"size": 8}
            )


plt.title("Ethereum Fraud Correlation: 'FLAG' Analysis (Row 1)", fontsize=25, pad=20)
plt.xticks(rotation=90)
plt.yticks(rotation=0)


plt.savefig('correlation_heatmap.png', bbox_inches='tight', dpi=300)
plt.show()

print("\n" + "="*40)
print("TOP 15 FEATURES FOR MODELS")
print("="*40)
top_correlations = corr_matrix['FLAG'].abs().sort_values(ascending=False)
print(top_correlations.head(16))