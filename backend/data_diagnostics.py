import pandas as pd
import numpy as np


def dataset_health_check(df: pd.DataFrame, target_col: str = None, threshold: float = 0.3):
    numeric_df = df.select_dtypes(include=["number"])

    # 1. NaN analysis
    nan_ratio = numeric_df.isna().mean().mean()

    # 2. Placeholder detection (AirQuality-safe)
    placeholder_values = [-200, 9999, -999]
    placeholder_mask = numeric_df.isin(placeholder_values)
    placeholder_ratio = placeholder_mask.sum().sum() / numeric_df.size

    # Per-column placeholder dominance
    placeholder_cols = (
        placeholder_mask.mean() > 0.3
    )
    bad_placeholder_cols = placeholder_cols[placeholder_cols].index.tolist()

    # 3. Low-information features (IQR-based)
    low_info_features = []
    for col in numeric_df.columns:
        q75, q25 = np.percentile(numeric_df[col].dropna(), [75, 25])
        iqr = q75 - q25
        if iqr < 1e-3:
            low_info_features.append(col)

    low_info_ratio = len(low_info_features) / max(1, len(numeric_df.columns))

    # 4. Target sanity
    target_nan_ratio, target_unique = None, None
    if target_col and target_col in df.columns:
        target_nan_ratio = df[target_col].isna().mean()
        target_unique = df[target_col].nunique()

    # 5. Row survival ratio
    row_survival_ratio = df.dropna().shape[0] / df.shape[0]

    # 6. Rating logic (stricter & honest)
    if (
        nan_ratio > threshold or
        placeholder_ratio > threshold or
        len(bad_placeholder_cols) > 0.3 * len(numeric_df.columns) or
        (target_nan_ratio and target_nan_ratio > 0.5) or
        (target_unique and target_unique <= 2)
    ):
        rating = "❌ Not fit for training"
    elif (
        placeholder_ratio > 0.1 or
        low_info_ratio > 0.3
    ):
        rating = "⚠️ Train with caution"
    else:
        rating = "✅ Fit for training"

    return {
        "nan_ratio": nan_ratio,
        "placeholder_ratio": placeholder_ratio,
        "bad_placeholder_cols": bad_placeholder_cols,
        "low_info_features": low_info_features,
        "low_info_ratio": low_info_ratio,
        "target_nan_ratio": target_nan_ratio,
        "target_unique": target_unique,
        "row_survival_ratio": row_survival_ratio,
        "rating": rating
    }
