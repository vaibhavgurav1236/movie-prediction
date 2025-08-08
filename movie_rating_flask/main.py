import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (ensure filename matches exactly)
df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')

# Drop rows where Rating is missing (our target variable)
df = df.dropna(subset=["Rating"])

# Fill missing values for important features
df["Genre"] = df["Genre"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Unknown")
df["Actor 1"] = df["Actor 1"].fillna("Unknown")

# Select relevant columns
df = df[["Genre", "Director", "Actor 1", "Rating"]]

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(df[["Genre", "Director", "Actor 1"]])
y = df["Rating"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Compatible with all sklearn versions
r2 = r2_score(y_test, y_pred)

# Display results
print("✅ Movie Rating Prediction Model Results")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Optional: Plot rating distribution
sns.histplot(df["Rating"], bins=20)
plt.title("🎬 IMDb Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()
00