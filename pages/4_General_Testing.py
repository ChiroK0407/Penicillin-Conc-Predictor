import streamlit as st
import pandas as pd
from backend.model_utils import train_model, train_benchmark_model

st.title("General Testing Page")

uploaded = st.file_uploader("Upload any CSV dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())

    target_col = st.selectbox("Select target column", df.columns)

    model_choice = st.selectbox(
        "Select model to test",
        [
            "linear", "poly", "rbf",
            "linear_scratch", "poly_scratch", "rbf_scratch",
            "rf", "mlp", "xgb"
        ]
    )

    split_type = st.radio(
        "Choose data split strategy",
        ["Time-ordered split", "Random shuffle split"]
    )

    if st.button("Run Test"):
        with st.spinner("Training model on uploaded dataset..."):
            if model_choice in ["linear", "poly", "rbf",
                                "linear_scratch", "poly_scratch", "rbf_scratch"]:
                model, data, metrics, scaler = train_model(
                    df, target_col, model_choice, split_type=split_type
                )
            else:
                model, data, metrics, scaler = train_benchmark_model(
                    df, target_col, model_choice, split_type=split_type
                )

        st.success("Model trained successfully on uploaded dataset.")

        st.subheader("📊 Metrics")
        st.write(metrics)

        st.subheader("🔍 Sample Predictions")
        y_pred = model.predict(data["X_test_scaled"])
        preview_df = pd.DataFrame({
            "Actual": data["y_test"],
            "Predicted": y_pred
        })
        st.dataframe(preview_df.head(20))

        st.divider()
        st.markdown("### 🔄 Reset Page")
        if st.button("Refresh Page"):
            for key in list(st.session_state.keys()):
                st.session_state[key] = None
            st.rerun()
else:
    st.info("Upload a dataset to begin testing.")