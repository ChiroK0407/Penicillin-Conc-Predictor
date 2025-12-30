import streamlit as st
import numpy as np

def analyze_model_performance(metrics, model_type, df, target_col):
    """
    Analyze model performance and dataset quality.
    Displays warnings and interpretations directly in Streamlit.
    """

    r2 = metrics.get("R2", None)
    mae = metrics.get("MAE", None)
    mse = metrics.get("MSE", None)

    # --- Data quality checks ---
    if target_col in df.columns:
        # Check for placeholder values like -200
        if df[target_col].isin([-200]).sum() > 0:
            st.warning(f"⚠️ Target column '{target_col}' contains placeholder values (-200). "
                       "These should be cleaned before training.")
        # Check for missing values
        if df[target_col].isna().sum() > 0:
            st.warning(f"⚠️ Target column '{target_col}' has {df[target_col].isna().sum()} missing values. "
                       "Consider cleaning or imputing before training.")

    # --- Model interpretation ---
    st.markdown("### 🤖 Model Interpretation")

    if r2 is None:
        st.info("No R² score available for interpretation.")
        return

    if r2 < 0:
        st.error(f"🧠 The {model_type} model performs worse than a constant baseline (R² = {r2:.3f}). "
                 "This suggests poor feature-target correlation or data quality issues.")
    elif r2 < 0.5:
        st.warning(f"🧠 The {model_type} model has weak predictive power (R² = {r2:.3f}). "
                   "Consider feature engineering, auto-tuning, or cleaning the dataset.")
    elif r2 < 0.8:
        st.info(f"✅ The {model_type} model shows moderate fit (R² = {r2:.3f}). "
                "You may improve it with tuning or better features.")
    else:
        st.success(f"🎉 The {model_type} model shows strong fit (R² = {r2:.3f}). "
                   "Performance looks good, but always validate with residual plots.")

    # --- Residual diagnostic suggestion ---
    if mae and mse:
        st.caption(f"Residual check: MAE = {mae:.3f}, MSE = {mse:.3f}. "
                   "Large errors may indicate outliers or unclean data.")