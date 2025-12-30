import streamlit as st
import pandas as pd
from backend.model_utils import train_model, train_benchmark_model
from backend.analyze_helper import analyze_model_performance   # <-- your new helper

st.title("General Testing Page")

uploaded = st.file_uploader("Upload any CSV dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())

    target_col = st.selectbox("Select target column", df.columns)

    split_type = st.radio(
        "Choose data split strategy",
        ["Time-ordered split", "Random shuffle split"]
    )

    mode = st.radio(
        "Choose testing mode",
        ["Single Model Testing", "Multiple Model Comparison"]
    )

    # -----------------------------
    # Single Model Testing
    # -----------------------------
    if mode == "Single Model Testing":
        model_type = st.selectbox(
            "Select model to test",
            [
                "linear", "poly", "rbf",
                "linear_scratch", "poly_scratch", "rbf_scratch",
                "rf", "mlp", "xgb"
            ]
        )

        if st.button("Run Test"):
            with st.spinner("Training model on uploaded dataset..."):
                if model_type in ["linear", "poly", "rbf",
                                  "linear_scratch", "poly_scratch", "rbf_scratch"]:
                    model, data, metrics, scaler = train_model(
                        df, target_col, model_type, split_type=split_type
                    )
                else:
                    model, data, metrics, scaler = train_benchmark_model(
                        df, target_col, model_type, split_type=split_type
                    )

            st.success("Model trained successfully on uploaded dataset.")

            st.header("📊 Model Performance Metrics")
            st.table(pd.DataFrame([metrics]))

            # AI-style analysis
            analyze_model_performance(metrics, model_type, df, target_col)

            st.subheader("🔍 Sample Predictions")
            y_pred = model.predict(data["X_test_scaled"])
            preview_df = pd.DataFrame({
                "Actual": data["y_test"],
                "Predicted": y_pred
            })
            st.dataframe(preview_df.head(20))

            # Actual vs Predicted Line Plot
            st.header("📈 Actual vs Predicted")
            st.line_chart(pd.DataFrame({
                "Actual": data["y_test"],
                "Predicted": y_pred
            }))

            # Residuals Plot
            st.header("📉 Residuals Plot")
            residuals = data["y_test"] - y_pred
            residuals_df = pd.DataFrame({
                "Index": range(len(residuals)),
                "Residuals": residuals
            })
            st.line_chart(residuals_df.set_index("Index"))

            st.divider()
            st.markdown("### 🔄 Reset Page")
            if st.button("Refresh Page"):
                for key in list(st.session_state.keys()):
                    st.session_state[key] = None
                st.rerun()

    # -----------------------------
    # Multiple Model Comparison
    # -----------------------------
    elif mode == "Multiple Model Comparison":
        model_types = ["linear", "poly", "rbf"]
        results = []

        # Perform split once
        model, data, metrics, scaler = train_model(
            df, target_col, "linear", split_type=split_type
        )
        X_train_scaled, X_test_scaled, y_train, y_test = (
            data["X_train_scaled"], data["X_test_scaled"],
            data["y_train"], data["y_test"]
        )

        for mtype in model_types:
            model, _, metrics, _ = train_model(
                df, target_col, mtype, split_type=split_type
            )
            y_pred = model.predict(X_test_scaled)
            results.append({
                "Model": mtype,
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "y_test": y_test,
                "y_pred": y_pred
            })

        # Rank models by R2
        results_sorted = sorted(results, key=lambda x: x["R2"], reverse=True)
        metrics_table = pd.DataFrame([{
            "Rank": i+1,
            "Model": r["Model"],
            "MSE": r["MSE"],
            "MAE": r["MAE"],
            "R2": r["R2"]
        } for i, r in enumerate(results_sorted)])

        st.header("🏆 Model Comparison Table")
        st.table(metrics_table)

        # AI-style analysis for each model
        for r in results_sorted:
            analyze_model_performance(
                {"MSE": r["MSE"], "MAE": r["MAE"], "R2": r["R2"]},
                r["Model"],
                df,
                target_col
            )

        # Scatter plot overlay
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        y_test = results_sorted[0]["y_test"]  # use same test set
        ax.plot(y_test, y_test, color="black", linestyle="--", label="y = x")

        for r in results_sorted:
            ax.scatter(y_test, r["y_pred"], label=r["Model"], alpha=0.6)

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        st.pyplot(fig)

        st.divider()
        st.markdown("### 🔄 Reset Page")
        if st.button("Refresh Page"):
            for key in list(st.session_state.keys()):
                st.session_state[key] = None
            st.rerun()

else:
    st.info("Upload a dataset to begin testing.")