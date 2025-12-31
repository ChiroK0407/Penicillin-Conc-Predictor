# Page4_General_Testing.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from backend.model_utils import train_model, train_benchmark_model
from backend.data_diagnostics import dataset_health_check

# -------------------------------------------------
# Page config & session isolation
# -------------------------------------------------
st.set_page_config(page_title="General Dataset Testing", layout="wide")

PAGE_PREFIX = "p4_"
def _k(name: str) -> str:
    return PAGE_PREFIX + name

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🧪 General Dataset Testing")
st.markdown("""
Upload **any CSV dataset** to evaluate:
- Dataset health & readiness
- Model behavior on unfamiliar data
- Robustness of implemented models
""")

# -------------------------------------------------
# Upload
# -------------------------------------------------
uploaded = st.file_uploader("📂 Upload CSV Dataset", type=["csv"], key=_k("upload"))

if not uploaded:
    st.info("Upload a dataset to begin testing.")
    st.stop()

# -------------------------------------------------
# Load dataset once
# -------------------------------------------------
if _k("df") not in st.session_state:
    st.session_state[_k("df")] = pd.read_csv(uploaded)

df = st.session_state[_k("df")]

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Target selection
# -------------------------------------------------
target_col = st.selectbox(
    "🎯 Select target column",
    df.columns,
    key=_k("target")
)

# -------------------------------------------------
# Dataset Health Diagnostics (recompute on target change)
# -------------------------------------------------
st.divider()
st.header("🩺 Dataset Health Diagnostics")

if (
    _k("diagnostics") not in st.session_state
    or st.session_state.get(_k("diagnostics_target")) != target_col
):
    st.session_state[_k("diagnostics")] = dataset_health_check(df, target_col)
    st.session_state[_k("diagnostics_target")] = target_col

diagnostics = st.session_state[_k("diagnostics")]

# -------------------------------------------------
# Health summary table
# -------------------------------------------------
health_table = pd.DataFrame({
    "Metric": [
        "NaN Ratio",
        "Placeholder Ratio",
        "Row Survival Ratio",
        "Target NaN Ratio",
        "Target Unique Values"
    ],
    "Value": [
        diagnostics["nan_ratio"],
        diagnostics["placeholder_ratio"],
        diagnostics["row_survival_ratio"],
        diagnostics.get("target_nan_ratio"),
        diagnostics.get("target_unique")
    ]
})

st.subheader("📋 Health Summary")
st.table(health_table)

# -------------------------------------------------
# Low-information features
# -------------------------------------------------
low_info = diagnostics.get("low_info_features", [])
if low_info:
    st.subheader("⚠️ Low-Information Features")
    st.table(pd.DataFrame(
        {"Feature": low_info}
    ).reset_index(drop=True))
else:
    st.info("No low-information features detected.")

# -------------------------------------------------
# Placeholder-dominated columns
# -------------------------------------------------
bad_cols = diagnostics.get("bad_placeholder_cols", [])
if bad_cols:
    st.warning("Columns heavily contaminated by placeholder values:")
    st.table(pd.DataFrame(
        {"Column": bad_cols}
    ).reset_index(drop=True))

# -------------------------------------------------
# Dataset rating
# -------------------------------------------------
st.subheader("📊 Dataset Rating")
rating = diagnostics["rating"]

if rating.startswith("✅"):
    st.success(rating)
elif rating.startswith("⚠️"):
    st.warning(rating)
else:
    st.error(rating)

if not rating.startswith("✅"):
    with st.expander("ℹ️ Why model performance may be poor"):
        st.markdown("""
- High placeholder contamination (e.g. `-200`)
- Near-constant or low-information features
- Weak regression signal
- Time-dependent structure violated by random splitting

This reflects **dataset quality**, not a model bug.
""")

# -------------------------------------------------
# Proceed / Refresh controls
# -------------------------------------------------
st.divider()
st.subheader("➡️ Next Action")

col1, col2 = st.columns([2, 1])
with col1:
    proceed = st.button("▶️ Proceed with Training", key=_k("proceed"))
with col2:
    refresh = st.button("🔄 Refresh Dataset", key=_k("refresh"))

if refresh:
    for k in list(st.session_state.keys()):
        if k.startswith(PAGE_PREFIX):
            del st.session_state[k]
    st.rerun()

if rating.startswith("❌") or not proceed:
    st.stop()

# -------------------------------------------------
# Split strategy
# -------------------------------------------------
st.divider()
st.subheader("🔀 Data Split Strategy")

has_time_col = any(c.lower().startswith(("date", "time")) for c in df.columns)
if has_time_col:
    st.info("Time-related column detected → using time-ordered split.")
    split_type = "Time-ordered split"
else:
    split_type = st.radio(
        "Choose split method",
        ["Time-ordered split", "Random shuffle split"],
        key=_k("split")
    )

# -------------------------------------------------
# Mode selection
# -------------------------------------------------
st.subheader("🧠 Testing Mode")

mode = st.radio(
    "Select testing mode",
    ["Single Model Testing", "Multiple Model Comparison"],
    key=_k("mode")
)

# =================================================
# SINGLE MODEL TESTING
# =================================================
if mode == "Single Model Testing":

    model_type = st.selectbox(
        "Select model",
        [
            "linear", "poly", "rbf",
            "linear_scratch", "poly_scratch", "rbf_scratch",
            "rf", "mlp", "xgb"
        ],
        key=_k("model")
    )

    if st.button("🚀 Run Model Test", key=_k("run_single")):
        with st.spinner("Training model..."):
            if model_type in ["rf", "mlp", "xgb"]:
                model, data, metrics, _ = train_benchmark_model(
                    df, target_col, model_type, split_type=split_type
                )
            else:
                model, data, metrics, _ = train_model(
                    df, target_col, model_type, split_type=split_type
                )

        st.success("Model training completed.")

        st.header("📊 Performance Metrics")
        st.table(pd.DataFrame([metrics]))

        y_pred = model.predict(data["X_test_scaled"])

        preview = pd.DataFrame({
            "Actual": data["y_test"],
            "Predicted": y_pred
        })

        st.subheader("🔍 Prediction Preview")
        st.dataframe(preview.head(20))

        st.header("📈 Actual vs Predicted")
        st.line_chart(preview)

        st.header("📉 Residuals")
        residuals = data["y_test"] - y_pred
        st.line_chart(pd.DataFrame({"Residuals": residuals}))

# =================================================
# MULTIPLE MODEL COMPARISON
# =================================================
else:

    model_types = ["linear", "poly", "rbf"]
    results = []

    base_model, data, _, _ = train_model(
        df, target_col, "linear", split_type=split_type
    )

    X_test = data["X_test_scaled"]
    y_test = data["y_test"]

    for m in model_types:
        model, _, metrics, _ = train_model(
            df, target_col, m, split_type=split_type
        )
        y_pred = model.predict(X_test)
        results.append({
            "Model": m,
            "MSE": metrics["MSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "y_pred": y_pred
        })

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    st.header("🏆 Model Comparison")
    st.table(results_df[["Rank", "Model", "MSE", "MAE", "R2"]])

    fig, ax = plt.subplots()
    ax.plot(y_test, y_test, "--", color="black", label="Ideal")

    for _, r in results_df.iterrows():
        ax.scatter(y_test, r["y_pred"], alpha=0.6, label=r["Model"])

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    st.pyplot(fig)
