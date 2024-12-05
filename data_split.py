import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Load the CSV file
df = pd.read_csv('dataset.csv')

# Step 2: Drop specified columns and columns with NaN values
columns_to_drop = ["Unnamed: 0", "ID"]
df_cleaned = df.drop(columns=columns_to_drop).dropna(axis=1)

# Step 3: Change the "Stage" column values
if 'Stage' in df_cleaned.columns:
    df_cleaned['Stage'] = df_cleaned['Stage'].astype(int) - 1

# Step 4: Split the cleaned dataset into train, test, and validation sets
train, temp = train_test_split(df_cleaned, test_size=0.3, random_state=42)  # 70% train
test, validation = train_test_split(temp, test_size=0.5, random_state=42)   # 15% test, 15% validation

# Step 5: Save the datasets into new CSV files
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
validation.to_csv('validation.csv', index=False)

print(train.columns)
print(np.unique(train["Stage"]))