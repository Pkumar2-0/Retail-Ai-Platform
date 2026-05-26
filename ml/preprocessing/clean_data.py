import pandas as pd

# Load dataset
df = pd.read_csv("ml/datasets/raw/walmart.csv")

# Check missing values
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Convert date column
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Save cleaned dataset
df.to_csv("ml/datasets/staged/cleaned_walmart.csv", index=False)

print("Data cleaned successfully!")
print(df.head())