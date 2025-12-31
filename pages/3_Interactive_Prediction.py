import streamlit as st
import pandas as pd
import numpy as np
from backend.retrain_utils import retrain_with_selected_features, predict_with_inputs

st.title("Interactive Prediction")

# -----------------------------
# Session state initialization
# -----------------------------
for key in [
    "base_model", "base_data", "base_metrics", "base_scaler",
    "feature_importances", "selected_features",
    "reduced_model", "reduced_data", "reduced_scaler", "rmse_error"
]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "selected_features" else []

# -----------------------------
# Upload and configuration
# -----------------------------
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    target_col = st.selectbox("Select target column", df.columns)

    model_choice = st.selectbox(
        "Select model type",
        ["linear", "poly", "rbf", "linear_scratch", "poly_scratch", "rbf_scratch"]
    )

    split_type = st.radio(
        "Choose data split strategy",
        ["Time-ordered split", "Random shuffle split"]
    )

    st.caption(f"Target selected: {target_col}. Ensure column units are correct.")

    # -----------------------------
    # Stage 1 — Train baseline model
    # -----------------------------
    st.markdown("### Stage 1 — Train baseline model")
    if st.button("Train Model"):
        from backend.model_utils import train_model
        with st.spinner("Training baseline model..."):
            model, data, metrics, scaler = train_model(
                df, target_col, model_choice, split_type=split_type
            )

        from backend.model_utils_ip import wrap_scaled_array
        X_train_df = wrap_scaled_array(data["X_train_scaled"], df, target_col)

        st.session_state.base_model = model
        st.session_state.base_data = data
        st.session_state.base_metrics = metrics
        st.session_state.base_scaler = scaler

        # Extract feature importances
        if hasattr(model, "coef_"):
            importances = np.array(model.coef_).flatten()
        elif hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_).flatten()
        else:
            importances = np.var(data["X_train"], axis=0)

        # Always use the wrapped DataFrame’s column names
        feature_names = X_train_df.columns

        # Normalize and rank
        importances = np.abs(importances)
        importance_pct = 100 * importances / np.sum(importances)
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance (%)": importance_pct
        }).sort_values("Importance (%)", ascending=False).reset_index(drop=True)
        df_imp.insert(0, "Rank", np.arange(1, len(df_imp) + 1))

        st.session_state.feature_importances = df_imp
        st.success("Baseline training completed. Feature importances computed.")

    if st.session_state.feature_importances is not None:
        st.subheader("Feature importances (normalized)")
        st.table(st.session_state.feature_importances)

    # -----------------------------
    # Stage 2 — Select features & retrain reduced model
    # -----------------------------
    st.markdown("### Stage 2 — Select features and retrain reduced model")

    if st.session_state.feature_importances is None:
        st.info("Train the baseline model first to see feature importances.")
    else:
        default_top = st.session_state.feature_importances["Feature"].head(3).tolist()
        selected = st.multiselect(
            "Select features to use for reduced model",
            st.session_state.feature_importances["Feature"].tolist(),
            default=default_top,
        )
        st.session_state.selected_features = selected

        if st.button("Retrain with selected features"):
            if not st.session_state.selected_features:
                st.warning("Please select at least one feature before retraining.")
            else:
                with st.spinner("Retraining reduced model..."):
                    model, data, scaler, rmse = retrain_with_selected_features(
                        df, target_col, model_choice, split_type, st.session_state.selected_features
                    )
                    st.session_state.reduced_model = model
                    st.session_state.reduced_data = data
                    st.session_state.reduced_scaler = scaler
                    st.session_state.rmse_error = rmse
                st.success("Reduced model retrained successfully.")
                st.caption(f"Reduced model RMSE: {st.session_state.rmse_error:.4f}")

    # -----------------------------
    # Stage 3 — Manual input and prediction
    # -----------------------------
    st.markdown("### Stage 3 — Manual input and prediction")

    if st.session_state.reduced_model is None:
        st.info("Retrain the reduced model first to enable manual prediction.")
    else:
        input_values = {}
        for feat in st.session_state.selected_features:
            fmin = float(df[feat].min())
            fmax = float(df[feat].max())
            fmean = float(df[feat].mean())
            input_values[feat] = st.number_input(
                f"{feat} (range {fmin:.2f} – {fmax:.2f})",
                min_value=fmin,
                max_value=fmax,
                value=fmean,
                step=(fmax - fmin) / 100 if fmax > fmin else 0.01,
                format="%.2f"
            )

        if st.button("Predict Target"):
            y_pred, error = predict_with_inputs(
                st.session_state.reduced_model,
                st.session_state.reduced_scaler,
                st.session_state.selected_features,
                input_values,
                st.session_state.rmse_error,
            )
            st.success(f"Predicted {target_col} = {y_pred:.2f} ± {error:.2f}")

            if "time" in target_col.lower() and y_pred > 500:
                st.info("Prediction seems unusually high. Consider adjusting inputs or reviewing feature selection.")

            st.divider()
            st.markdown("### 🔄 Reset page")
            if st.button("Refresh Page"):
                for key in [
                    "base_model", "base_data", "base_metrics", "base_scaler",
                    "feature_importances", "selected_features",
                    "reduced_model", "reduced_data", "reduced_scaler", "rmse_error"
                ]:
                    st.session_state[key] = None if key != "selected_features" else []
                st.rerun()
else:
    st.info("Upload a CSV dataset to begin.")