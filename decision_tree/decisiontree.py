# ===============================
# STEP 1: Import Libraries
# ===============================
import pandas as pd
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ===============================
# STEP 2: Download Kaggle Dataset
# ===============================
print("=== DOWNLOADING COVID-19 DATASET FROM KAGGLE ===")
dataset_path = kagglehub.dataset_download("meirnizri/covid19-dataset")
files = os.listdir(dataset_path)
csv_file = [f for f in files if f.endswith('.csv')][0]  # pick first CSV
file_path = os.path.join(dataset_path, csv_file)

# ===============================
# STEP 3: Load CSV into DataFrame
# ===============================
df = pd.read_csv(file_path)
print("\n=== RAW DATASET (First 5 Rows) ===")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ===============================
# STEP 4: Preprocessing
# ===============================

# Example: dynamically select numeric columns
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# Pick a target column (for demonstration, we use the last numeric column)
# In real use, choose the column you want to predict
target_col = numeric_cols[-1]
feature_cols = [col for col in numeric_cols if col != target_col]

print(f"\nSelected Features: {feature_cols}")
print(f"Target Column: {target_col}")

# Drop rows with missing values for simplicity
df_clean = df[feature_cols + [target_col]].dropna()

X = df_clean[feature_cols]
y = df_clean[target_col]

# ===============================
# STEP 5: Split Train/Test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# STEP 6: Train Decision Tree
# ===============================
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# ===============================
# STEP 7: Predictions & Evaluation
# ===============================
y_pred = clf.predict(X_test)

print("\n=== MODEL EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# STEP 8: Visualize Decision Tree
# ===============================
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=feature_cols, filled=True, rounded=True)
plt.show()
