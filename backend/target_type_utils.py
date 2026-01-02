# ===================================
# target_type_utils.py
# ===================================
import pandas as pd
import numpy as np

def scan_dataset_targets(df: pd.DataFrame):
    """
    Scan entire dataset to detect possible target columns.
    Returns dict with detected targets categorized by type.
    
    Detection rules:
    - Binary: Exactly 2 unique values in {0, 1}
    - Multiclass: 3-10 unique integer values
    - Regression: >10 unique numeric values
    """
    binary_targets = []
    multiclass_targets = []
    regression_targets = []

    for col in df.columns:
        series = df[col].dropna()

        if series.empty:
            continue

        # Try numeric conversion
        try:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
        except:
            continue
        
        if numeric.empty:
            continue
            
        unique_vals = np.unique(numeric)
        n_unique = len(unique_vals)

        # Skip ID-like columns (by name heuristic + all unique values)
        if n_unique == len(series) and any(kw in col.lower() for kw in ['id', 'index', 'key', 'code', 'number', 'no']):
            print(f"Skipping '{col}' - appears to be an ID column")
            continue

        # Binary: exactly 2 values from {0, 1}
        if n_unique == 2 and set(unique_vals).issubset({0, 1}):
            binary_targets.append(col)
            print(f"Detected binary target: {col} with values {unique_vals}")

        # Multiclass: 3-10 unique integer values
        elif (
            pd.api.types.is_integer_dtype(numeric)
            and 2 < n_unique <= 10
        ):
            multiclass_targets.append(col)
            print(f"Detected multiclass target: {col} with {n_unique} classes")

        # Regression: more than 10 unique values
        elif n_unique > 10:
            regression_targets.append(col)
            print(f"Detected regression target: {col} with {n_unique} unique values")

    result = {
        "binary": binary_targets,
        "multiclass": multiclass_targets,
        "regression": regression_targets
    }
    
    print(f"\nTarget detection summary:")
    print(f"  Binary: {len(binary_targets)}")
    print(f"  Multiclass: {len(multiclass_targets)}")
    print(f"  Regression: {len(regression_targets)}")
    
    return result