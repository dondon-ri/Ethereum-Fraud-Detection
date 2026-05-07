#checking the dataset balance
import pandas as pd

# Load the data
df = pd.read_csv("transaction_dataset.csv")

#the raw counts
print(df['FLAG'].value_counts())

#percentages
print(df['FLAG'].value_counts(normalize=True) * 100)