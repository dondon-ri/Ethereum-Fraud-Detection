import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_clean = pd.read_csv('cleaned_ethereum_data.csv')

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