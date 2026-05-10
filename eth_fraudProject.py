import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import minmax_scale
df = pd.read_csv("transaction_dataset.csv") #original columns -> 51

#cleaning 3 columns data
useless_col = ['Unnamed: 0', 'Index', 'Address']
df_cleaned = df.drop(columns=useless_col)
print(f"Original Columns: {df.shape[1]}")
print(f"Cleaned Columns: {df_cleaned.shape[1]}")

#Handle missing value
df_cleaned = df_cleaned.fillna(0)
print("Missing values after cleaning:")
print(df_cleaned.isnull().sum().sum()) # result = 0 mean there is no missing value

#Encoding
le = LabelEncoder()
cat_cols = [' ERC20 most sent token type', ' ERC20_most_rec_token_type']
for col in cat_cols:
    df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
print("Categorical encoding finished.")

#scaling
scaler = MinMaxScaler()
X = df_cleaned.drop('FLAG', axis=1)
y = df_cleaned['FLAG']
X_scaled = scaler.fit_transform(X)
eth_final = pd.DataFrame(X_scaled,columns=X.columns)
df['Flag'] = y.values
print("Data Scaling finished. Ready for modeling!")
print(eth_final.head())

#to save as cleaned_data
final_df = eth_final.copy()
final_df['FLAG'] = y.values
final_df.to_csv("cleaned_ethereum_data.csv", index=False)