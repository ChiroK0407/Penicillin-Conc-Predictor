import pandas as pd

def wrap_scaled_array(X_scaled, original_df, target_col):
    """
    Wraps a scaled NumPy array into a DataFrame using original column names
    after preprocessing. Assumes original_df is the raw DataFrame before scaling.

    Parameters:
    - X_scaled: np.ndarray (output from scaler)
    - original_df: pd.DataFrame (raw input before preprocessing)
    - target_col: str (name of target column to exclude)

    Returns:
    - pd.DataFrame with same column names as original_df minus target_col
    """
    # Drop target column and non-numeric columns
    X_raw = original_df.drop(columns=[target_col])
    X_raw = X_raw.select_dtypes(include=["number"])
    X_raw = X_raw.dropna(axis=1, how="all")

    # Align column names to scaled array
    return pd.DataFrame(X_scaled, columns=X_raw.columns)