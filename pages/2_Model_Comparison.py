import streamlit as st
import pandas as pd
from backend.model_utils import (
    train_model, autotune_model,
    train_benchmark_model, autotune_benchmark_model
)
from backend.plotting import overlay_plot, multi_plot, multi_scatter_plot
# Import scratch models
from backend.models.linear_svr import LinearSVR_Scratch
from backend.models.poly_svr import PolySVR_Scratch
from backend.models.rbf_svr import RBFSVRScratch

st.title("Model Comparison")

uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

# State
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = {}
if "tuned_results" not in st.session_state:
    st.session_state.tuned_results = None

if uploaded:
    df = pd.read_csv(uploaded)
    target_col = st.selectbox("Select target column", df.columns)

    # Choose comparison mode
    mode = st.radio(
        "Select comparison mode",
        ["Compare Different Models", "Compare Tuned vs Untuned (Single Model)"]
    )

    split_type = st.radio(
        "Choose data split strategy",
        ["Time-ordered split", "Random shuffle split"]
    )

    # --- Mode 1: Compare Different Models ---
    if mode == "Compare Different Models":
        model_choices = st.multiselect(
            "Select models to compare",
            [
                "linear", "poly", "rbf",
                "linear_scratch", "poly_scratch", "rbf_scratch",
                "rf", "mlp", "xgb"
            ]
        )

        if st.button("Run Comparison"):
            results = {}
            for m in model_choices:
                if m in ["linear", "poly", "rbf",
                         "linear_scratch", "poly_scratch", "rbf_scratch"]:
                    model, data, metrics, scaler = train_model(
                        df, target_col, m, split_type=split_type
                    )
                else:  # benchmark models
                    model, data, metrics, scaler = train_benchmark_model(
                        df, target_col, m, split_type=split_type
                    )
                y_pred = model.predict(data["X_test_scaled"]) \
                    if hasattr(model, "predict") else None
                results[m] = {
                    "model": model,
                    "data": data,
                    "metrics": metrics,
                    "y_pred": y_pred
                }
            st.session_state.comparison_results = results

        if st.session_state.comparison_results:
            results = st.session_state.comparison_results

            st.header("📊 Metrics Comparison")
            metrics_df = pd.DataFrame({
                m: res["metrics"] for m, res in results.items()
            })
            st.table(metrics_df)

            st.header("📈 Overlay Plot: Multiple Models")
            predictions_dict = {m: res["y_pred"] for m, res in results.items()}
            first_res = next(iter(results.values()))
            fig = multi_plot(
                first_res["data"]["X_test"],
                first_res["data"]["y_test"],
                predictions_dict,
                target_col
            )
            st.pyplot(fig)

            st.header("📊 Scatter Plot: Actual vs Predicted")
            fig2 = multi_scatter_plot(
                first_res["data"]["y_test"],
                predictions_dict,
                target_col
            )
            st.pyplot(fig2)

            st.divider()
            st.markdown("### 🔄 Reset Comparison")
            if st.button("Refresh Page"):
                st.session_state.comparison_results = {}
                st.rerun()

    # --- Mode 2: Compare Tuned vs Untuned ---
    elif mode == "Compare Tuned vs Untuned (Single Model)":
        model_choice = st.selectbox(
            "Select a single model",
            [
                "linear", "poly", "rbf",
                "linear_scratch", "poly_scratch", "rbf_scratch",
                "rf", "mlp", "xgb"
            ]
        )

        if st.button("Run Tuned vs Untuned Comparison"):
            # Train untuned
            if model_choice in ["linear", "poly", "rbf",
                                "linear_scratch", "poly_scratch", "rbf_scratch"]:
                model, data, metrics, scaler = train_model(
                    df, target_col, model_choice, split_type=split_type
                )
            else:
                model, data, metrics, scaler = train_benchmark_model(
                    df, target_col, model_choice, split_type=split_type
                )
            y_pred = model.predict(data["X_test_scaled"])

            # Run autotune
            if model_choice in ["linear", "poly", "rbf",
                                "linear_scratch", "poly_scratch", "rbf_scratch"]:
                best_params, best_metrics, best_score = autotune_model(
                    df, target_col, model_choice, split_type=split_type
                )
            else:
                best_params, best_metrics, best_score = autotune_benchmark_model(
                    df, target_col, model_choice, split_type=split_type
                )

            if best_params is None:
                st.warning("Auto-tune not available for this model type.")
            else:
                # Build tuned model
                if model_choice in ["linear", "poly", "rbf", "rf", "mlp", "xgb"]:
                    tuned_model = type(model)(**best_params)
                elif model_choice == "linear_scratch":
                    tuned_model = LinearSVR_Scratch(**best_params)
                elif model_choice == "poly_scratch":
                    tuned_model = PolySVR_Scratch(**best_params)
                elif model_choice == "rbf_scratch":
                    tuned_model = RBFSVRScratch(**best_params)

                tuned_model.fit(data["X_train_scaled"], data["y_train"])
                y_pred_tuned = tuned_model.predict(data["X_test_scaled"])

                st.session_state.tuned_results = {
                    "untuned_metrics": metrics,
                    "tuned_metrics": best_metrics,
                    "y_pred_untuned": y_pred,
                    "y_pred_tuned": y_pred_tuned,
                    "data": data
                }

        if st.session_state.tuned_results:
            res = st.session_state.tuned_results

            st.header("📊 Metrics Comparison: Untuned vs Tuned")
            metrics_comparison = pd.DataFrame({
                "Untuned": res["untuned_metrics"],
                "Tuned": res["tuned_metrics"]
            })
            st.table(metrics_comparison)

            st.header("📈 Overlay Plot: Untuned vs Tuned Predictions")
            fig = overlay_plot(
                res["data"]["X_test"],
                res["data"]["y_test"],
                res["y_pred_untuned"],
                res["y_pred_tuned"],
                target_col
            )
            st.pyplot(fig)

            st.header("📊 Scatter Plot: Actual vs Predicted")
            fig2 = multi_scatter_plot(
                res["data"]["y_test"],
                {
                    "Untuned": res["y_pred_untuned"],
                    "Tuned": res["y_pred_tuned"]
                },
                target_col
            )
            st.pyplot(fig2)

            st.divider()
            st.markdown("### 🔄 Reset Comparison")
            if st.button("Refresh Page"):
                st.session_state.tuned_results = None
                st.rerun()