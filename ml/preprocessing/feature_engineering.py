import pandas as pd

# Load cleaned dataset
df = pd.read_csv("ml/datasets/staged/cleaned_walmart.csv")

# Convert Date column again
df["Date"] = pd.to_datetime(df["Date"])

# Create new features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Week"] = df["Date"].dt.isocalendar().week

# Save engineered dataset
df.to_csv("ml/datasets/curated/featured_walmart.csv", index=False)

print("Feature engineering completed successfully!")
print(df.head())