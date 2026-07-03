import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path

# Find project root based on current file location
current_file = Path(__file__)  
project_root = current_file.parent.parent.parent 

file1_path = project_root / 'sachs/Data Files/cd3cd28.csv'
file2_path = project_root / 'sachs/Data Files/cd3cd28_u0126.csv'

def apply_log_transform(data, epsilon=1e-6):
    """
    Apply log transformation to continuous variables
    epsilon: Small value for handling 0 or negative values
    """
    # Check minimum value to handle negative or zero values
    min_val = data.min()
    if min_val <= 0:
        # If minimum value is 0 or less, add constant to make all values positive
        shift_value = abs(min_val) + epsilon
        data_shifted = data + shift_value
        print(f"  Negative/0 value detected. Applied shift of {shift_value:.6f}")
    else:
        data_shifted = data
    
    # Apply log transformation
    log_transformed = np.log(data_shifted)
    return log_transformed

# Process first file
print("=== Processing cd3cd28.csv ===")
data1 = pd.read_csv(file1_path)
print(f"Original column names: {list(data1.columns)}")
print(f"Original data shape: {data1.shape}")

# Reorder columns: 0: PKC, 1: Plcg, 2: PKA, 3: PIP3, 4: P38, 5: Raf, 6: Jnk, 7: PIP2, 8: Mek, 9: Erk, 10: Akt
# Original order: ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"]
# Indices:        [0,     1,     2,      3,      4,      5,     6,     7,     8,     9,     10]
# Original indices for required order: [8, 2, 7, 4, 9, 0, 10, 3, 1, 5, 6]

reordered_data1 = data1.iloc[:, [8, 2, 7, 4, 9, 0, 10, 3, 1, 5, 6]]
numpy_array1 = reordered_data1.values.astype(float)

print(f"Statistics before transformation:")
print(f"  Mean: {numpy_array1.mean(axis=0)}")
print(f"  Std: {numpy_array1.std(axis=0)}")

# Apply log transformation
print("\n=== Applying Log Transformation ===")
column_names = ["PKC", "Plcg", "PKA", "PIP3", "P38", "Raf", "Jnk", "PIP2", "Mek", "Erk", "Akt"]
log_transformed_array1 = np.zeros_like(numpy_array1)

for i, col_name in enumerate(column_names):
    print(f"  Transforming {col_name} with log...")
    log_transformed_array1[:, i] = apply_log_transform(numpy_array1[:, i])

print(f"Statistics after log transformation:")
print(f"  Mean: {log_transformed_array1.mean(axis=0)}")
print(f"  Std: {log_transformed_array1.std(axis=0)}")

# Process second file
print("\n=== Processing cd3cd28_u0126.csv ===")
data2 = pd.read_csv(file2_path)
print(f"Original column names: {list(data2.columns)}")
print(f"Original data shape: {data2.shape}")

# Reorder columns in the same way
reordered_data2 = data2.iloc[:, [8, 2, 7, 4, 9, 0, 10, 3, 1, 5, 6]]
numpy_array2 = reordered_data2.values.astype(float)

print(f"Statistics before transformation:")
print(f"  Mean: {numpy_array2.mean(axis=0)}")
print(f"  Std: {numpy_array2.std(axis=0)}")

# Apply log transformation
print("\n=== Applying Log Transformation ===")
log_transformed_array2 = np.zeros_like(numpy_array2)

for i, col_name in enumerate(column_names):
    print(f"  Transforming {col_name} with log...")
    log_transformed_array2[:, i] = apply_log_transform(numpy_array2[:, i])

print(f"Statistics after log transformation:")
print(f"  Mean: {log_transformed_array2.mean(axis=0)}")
print(f"  Std: {log_transformed_array2.std(axis=0)}")

# Perform standardization (on log-transformed data)
print("\n=== Performing Standardization ===")

# Standardize each environment independently
scaler1 = StandardScaler()
normalized_array1 = scaler1.fit_transform(log_transformed_array1)

scaler2 = StandardScaler()
normalized_array2 = scaler2.fit_transform(log_transformed_array2)

print(f"Environment 1 after standardization:")
print(f"  Mean: {normalized_array1.mean(axis=0)}")
print(f"  Std: {normalized_array1.std(axis=0)}")

print(f"Environment 2 after standardization:")
print(f"  Mean: {normalized_array2.mean(axis=0)}")
print(f"  Std: {normalized_array2.std(axis=0)}")

# Save
np.save('/home/statduck/causal-flows/sachs/data_env1.npy', normalized_array1)
print(f"\ndata_env1.npy saved - shape: {normalized_array1.shape}")
print(f"First 5 rows:\n{normalized_array1[:5]}")

np.save('/home/statduck/causal-flows/sachs/data_env2.npy', normalized_array2)
print(f"\ndata_env2.npy saved - shape: {normalized_array2.shape}")
print(f"First 5 rows:\n{normalized_array2[:5]}")

print("\n=== Processing Complete ===")
print("Processing steps: 1) Column reordering → 2) Log transformation → 3) Standardization")
print("New column order: 0: PKC, 1: Plcg, 2: PKA, 3: PIP3, 4: P38, 5: Raf, 6: Jnk, 7: PIP2, 8: Mek, 9: Erk, 10: Akt")
print("Data has been log-transformed and standardized (mean=0, std=1)")
print("Standardization parameters have also been saved separately.")

# Summary of standardization information (based on log-transformed data)
print("\n=== Standardization Parameters Summary (After Log Transformation) ===")
print("Environment 1 (cd3cd28):")
for i, col in enumerate(column_names):
    print(f"  {col}: mean={scaler1.mean_[i]:.3f}, std={scaler1.scale_[i]:.3f}")

print("\nEnvironment 2 (cd3cd28_u0126):")
for i, col in enumerate(column_names):
    print(f"  {col}: mean={scaler2.mean_[i]:.3f}, std={scaler2.scale_[i]:.3f}")