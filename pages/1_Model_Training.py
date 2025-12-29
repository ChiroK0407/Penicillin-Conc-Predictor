import streamlit as st
import pandas as pd
from backend.model_utils import train_model, autotune_model
from backend.plotting import overlay_plot
from backend.diagnostics import run_diagnostics
from backend.models.linear_svr import LinearSVR_Scratch
from backend.models.poly_svr import PolySVR_Scratch
from backend.models.rbf_svr import RBFSVRScratch

# Inject CSS for sticky footer
st.markdown(
    """
    <style>
    .footer-buttons {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 10px 0;
        border-top: 1px solid #ddd;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Model Training & Auto-tuning")

uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
model_type = st.selectbox(
    "Choose model type",
    ["linear","poly","rbf","linear_scratch","poly_scratch","rbf_scratch"]
)

# State variables
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "autotune_result" not in st.session_state:
    st.session_state.autotune_result = None

if uploaded:
    df = pd.read_csv(uploaded)
    target_col = st.selectbox("Select target column", df.columns)

    # Split toggle
    split_type = st.radio(
        "Choose data split strategy",
        ["Time-ordered split", "Random shuffle split"]
    )

    # Train button
    if st.button("Train Model"):
        model, data, metrics, scaler = train_model(
            df, target_col, model_type, split_type=split_type
        )
        st.session_state.train_result = {
            "model": model,
            "data": data,
            "metrics": metrics,
            "scaler": scaler,
            "target_col": target_col,
            "split_type": split_type
        }
        st.session_state.autotune_result = None

    # Show training results
    if st.session_state.train_result:
        res = st.session_state.train_result

        st.header("📊 Model Performance Metrics (Test Set)")
        st.table(pd.DataFrame([res["metrics"]]))

        st.header("📈 Actual vs Predicted Curve (Original Model)")
        st.line_chart(
            pd.DataFrame({
                "Actual": res["data"]["y_test"],
                "Predicted": res["model"].predict(res["data"]["X_test_scaled"])
            })
        )

        # Show autotune results if available
        if st.session_state.autotune_result:
            auto = st.session_state.autotune_result

            st.header("⚙️ Auto-tuned Hyperparameters")
            st.table(pd.DataFrame([auto["best_params"]]))

            st.header("📊 Metrics Comparison: Original vs Auto-tuned")
            metrics_comparison = pd.DataFrame({
                "Original": res["metrics"],
                "Auto-tuned": auto["best_metrics"]
            })
            st.table(metrics_comparison)

            st.header("📈 Overlay Plot: Actual vs Original vs Auto-tuned Predictions")
            fig = overlay_plot(
                res["data"]["X_test"],
                res["data"]["y_test"],
                res["model"].predict(res["data"]["X_test_scaled"]),
                auto["y_pred_tuned"],
                res["target_col"]
            )
            st.pyplot(fig)

        # --- Actions always at bottom ---
        st.markdown('<div class="footer-buttons">', unsafe_allow_html=True)
        st.divider()
        st.markdown("### 🔧 Actions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            run_auto = st.button("Run Auto-tune", key="auto_button")
        with col2:
            run_diag = st.button("Run Diagnostics", key="diag_button")
        with col3:
            refresh = st.button("Refresh Page", key="refresh_button")

        st.markdown('</div>', unsafe_allow_html=True)

        # Auto-tune logic
        if run_auto:
            with st.spinner("Running lightweight auto-tune..."):
                best_params, best_metrics, best_score = autotune_model(
                    df, res["target_col"], model_type, split_type=res["split_type"]
                )
                if best_params is None:
                    st.warning("Auto-tune not available for this model type.")
                else:
                    if model_type in ["linear","poly","rbf"]:
                        tuned_model = type(res["model"])(**best_params)
                    elif model_type == "linear_scratch":
                        tuned_model = LinearSVR_Scratch(**best_params)
                    elif model_type == "poly_scratch":
                        tuned_model = PolySVR_Scratch(**best_params)
                    elif model_type == "rbf_scratch":
                        tuned_model = RBFSVRScratch(**best_params)

                    tuned_model.fit(res["data"]["X_train_scaled"], res["data"]["y_train"])
                    y_pred_tuned = tuned_model.predict(res["data"]["X_test_scaled"])

                    st.session_state.autotune_result = {
                        "best_params": best_params,
                        "best_metrics": best_metrics,
                        "best_score": best_score,
                        "y_pred_tuned": y_pred_tuned
                    }

        # Diagnostics logic
        if run_diag:
            run_diagnostics(res["model"], res["data"], df, res["target_col"])

        # Refresh logic
        if refresh:
            st.session_state.train_result = None
            st.session_state.autotune_result = None
            st.rerun()

