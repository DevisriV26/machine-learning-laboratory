import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt   # Uncomment if matplotlib works on your system

# Dataset: 5 students with 4 attributes
data = {
    'Internal': [78,85,72,90,75],
    'Attendance': [85,90,80,95,82],
    'Assignment': [80,88,75,92,78],
    'Quiz': [75,82,70,88,73]
}
df = pd.DataFrame(data)

print("RAW DATASET")
print(df)

# Mean and standard deviation
mean = df.mean()
std = df.std(ddof=1)

print("\nMEAN OF EACH ATTRIBUTE")
print(mean)
print("\nSTANDARD DEVIATION")
print(std)

# Standardize dataset (Z-score)
Z = (df - mean) / std
print("\nSTANDARDIZED DATA (Z-SCORES)")
print(Z)

# Covariance matrix
cov_matrix = np.cov(Z.T)
print("\nCOVARIANCE MATRIX")
print(cov_matrix)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEIGENVALUES")
print(eigenvalues)
print("\nEIGENVECTORS")
print(eigenvectors)

# Explained variance
explained_variance = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance)

print("\nEXPLAINED VARIANCE (IN %)")
print(explained_variance * 100)
print("\nCUMULATIVE VARIANCE (IN %)")
print(cumulative_variance * 100)

# Optional: Scree Plot
"""
import matplotlib.pyplot as plt
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.title("Scree Plot")
plt.grid()
plt.show()
"""

# Principal Component Scores
PC_scores = Z.dot(eigenvectors)
print("\nPRINCIPAL COMPONENT SCORES")
print(PC_scores)

# Reduce dataset to 2 principal components (PC1 and PC2)
PC_reduced = PC_scores.iloc[:, :2]
print("\nREDUCED DATASET (USING PC1 AND PC2)")
print(PC_reduced)
