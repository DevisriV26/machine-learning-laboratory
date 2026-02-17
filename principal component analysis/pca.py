import numpy as np
import pandas as pd
import os
import kagglehub

# ==============================
# STEP 1: Download Dataset
# ==============================
path = kagglehub.dataset_download(
    "sadiajavedd/students-academic-performance-dataset"
)

# Locate CSV file
files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
file_path = os.path.join(path, csv_file)


# ==============================
# STEP 2: Encode Categorical Columns
# ==============================
df_encoded = pd.get_dummies(df, drop_first=True)

# ==============================
# STEP 3: Calculate Mean
# ==============================
mean = df_encoded.mean()
print("\nMEAN OF EACH ATTRIBUTE")
print(mean)

# ==============================
# STEP 4: Standardize Data (Z-score)
# ==============================
std = df_encoded.std(ddof=1)
Z = (df_encoded - mean) / std

# ==============================
# STEP 5: Covariance Matrix
# ==============================
cov_matrix = np.cov(Z.T)
print("\nCOVARIANCE MATRIX")
print(cov_matrix)

# ==============================
# STEP 6: Eigenvalues & Eigenvectors
# ==============================
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues & eigenvectors in descending order
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

print("\nEIGENVALUES")
print(eigenvalues)

print("\nEIGENVECTORS")
print(eigenvectors)

# ==============================
# STEP 7: Principal Component Scores
# ==============================
PC_scores = np.dot(Z.values, eigenvectors)

# Convert to DataFrame for readability
PC_scores = pd.DataFrame(
    PC_scores,
    columns=[f'PC{i+1}' for i in range(len(eigenvalues))]
)

print("\nPRINCIPAL COMPONENT SCORES (First 5 Rows)")
print(PC_scores.head())

# ==============================
# STEP 8: Optional - First 3 PCs Only
# ==============================
PC_top3 = PC_scores[['PC1','PC2','PC3']]
print("\nFIRST 3 PRINCIPAL COMPONENTS (First 5 Rows)")
print(PC_top3)

