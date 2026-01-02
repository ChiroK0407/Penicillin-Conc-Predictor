import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer  # For NaN handling
from sklearn.metrics import confusion_matrix

from backend.model_utils import train_model, train_benchmark_model
from backend.data_diagnostics import dataset_health_check
from backend.target_type_utils import scan_dataset_targets

# -------------------------------------------------
# Page config
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
# Load dataset once (with full state clear on new upload)
# -------------------------------------------------
if _k("df") not in st.session_state or st.session_state.get(_k("file_name")) != uploaded.name:
    df = pd.read_csv(uploaded)
    st.session_state[_k("df")] = df
    st.session_state[_k("file_name")] = uploaded.name
    # Clear all other session state when new file is uploaded
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(PAGE_PREFIX) and k not in [_k("df"), _k("file_name")]]
    for k in keys_to_remove:
        if k in st.session_state:
            del st.session_state[k]
else:
    df = st.session_state[_k("df")]

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Display basic info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", len(df))
with col2:
    st.metric("Columns", len(df.columns))
with col3:
    st.metric("Numeric Columns", len(df.select_dtypes(include=["number"]).columns))

# -------------------------------------------------
# Auto target detection (with Regression Priority)
# -------------------------------------------------
st.divider()
st.subheader("🎯 Target Detection")

targets = scan_dataset_targets(df)

st.write("**Detected Targets:**")
if targets["binary"]:
    st.success(f"✅ Binary: {', '.join(targets['binary'])}")
if targets["multiclass"]:
    st.info(f"ℹ️ Multiclass: {', '.join(targets['multiclass'])}")
if targets["regression"]:
    st.info(f"📊 Regression: {', '.join(targets['regression'])}")

# Priority: Regression > Binary > Multiclass (as suggested)
all_targets = targets["regression"] + targets["binary"] + targets["multiclass"]

if not all_targets:
    st.error("❌ No suitable target column detected.")
    st.info("**Hints:**\n- Binary targets should have values {0, 1}\n- Multiclass targets should have 3-10 unique integer values\n- Regression targets should have >10 unique numeric values")
    st.stop()

# Auto-select based on priority
detected_col = all_targets[0]  # First in priority order
detected_type = "regression" if detected_col in targets["regression"] else "binary" if detected_col in targets["binary"] else "multiclass"

# Safety Switch: User Override
col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("Select Target", options=df.columns, index=list(df.columns).index(detected_col), key=_k("target_select"))
with col2:
    final_type = st.radio("Task Type", ["regression", "classification"], 
                          index=0 if detected_type == "regression" else 1, key=_k("task_type"))

# Confirm type based on user choice
if final_type == "regression":
    target_type = "regression"
    st.success(f"**Selected:** `{target_col}` (Regression)")
elif final_type == "classification":
    target_type = "binary" if df[target_col].nunique() == 2 else "multiclass"
    st.success(f"**Selected:** `{target_col}` (Classification - {target_type})")
else:
    st.error("Invalid task type selected.")
    st.stop()

# -------------------------------------------------
# Dataset Health Diagnostics (stricter thresholds)
# -------------------------------------------------
st.divider()
st.header("🩺 Dataset Health Diagnostics")

if (
    _k("diagnostics") not in st.session_state
    or st.session_state.get(_k("diagnostics_target")) != target_col
):
    with st.spinner("Running health diagnostics..."):
        st.session_state[_k("diagnostics")] = dataset_health_check(df, target_col)
        st.session_state[_k("diagnostics_target")] = target_col

diagnostics = st.session_state[_k("diagnostics")]

nan_ratio = diagnostics.get('nan_ratio', 0)
placeholder_ratio = diagnostics.get('placeholder_ratio', 0)
target_nan_ratio = diagnostics.get('target_nan_ratio', 0)
row_survival_ratio = diagnostics.get('row_survival_ratio', 1.0)
target_unique = diagnostics.get('target_unique', 0)

pipeline_type = "classification" if target_type in ["binary", "multiclass"] else "regression"

# Stricter rating (placeholders >5% = not suggested)
if target_nan_ratio > 0.05 or placeholder_ratio > 0.05:
    rating = "❌ Training not suggested"
    rating_emoji = "❌"
    reason = "due to high placeholders or target issues"
elif nan_ratio > 0.1 or row_survival_ratio < 0.7:
    rating = "⚠️ Proceed with caution"
    rating_emoji = "⚠️"
    reason = ""
else:
    rating = "✅ Fit"
    rating_emoji = "✅"
    reason = ""

display_rating = f"{rating} for {pipeline_type} type training{': ' + reason if reason else ''}"

is_regression_suitable = 1 if target_type == "regression" and rating_emoji == "✅" else 0

# Health table
health_data = {
    "Metric": ["Missing Values (NaN)", "Placeholder Values", "Row Survival Rate", "Target Missing Values", "Target Unique Values"],
    "Value": [f"{nan_ratio:.2%}", f"{placeholder_ratio:.2%}", f"{row_survival_ratio:.2%}", f"{target_nan_ratio:.2%}", str(target_unique)]
}
health_table = pd.DataFrame(health_data)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📋 Health Summary")
    st.table(health_table)
with col2:
    st.subheader("📊 Dataset Rating")
    if rating_emoji == "✅":
        st.success(display_rating)
    elif rating_emoji == "⚠️":
        st.warning(display_rating)
    else:
        st.error(display_rating)
    
    issues = []
    if nan_ratio > 0.05: issues.append(f"- Missing values ({nan_ratio:.1%})")
    if placeholder_ratio > 0.05: issues.append(f"- High placeholders ({placeholder_ratio:.1%})")
    if target_nan_ratio > 0: issues.append(f"- Target missing ({target_nan_ratio:.1%})")
    if row_survival_ratio < 0.8: issues.append(f"- Low survival ({row_survival_ratio:.1%})")
    
    if issues:
        st.warning("**Issues:**\n" + "\n".join(issues))
    elif rating_emoji == "✅":
        st.success("✅ No major issues!")

# Imputation (if needed)
if nan_ratio > 0.05 or placeholder_ratio > 0.05:
    impute_strategy = st.selectbox("Handle Issues?", ["None", "Mean", "Median"], key=_k("impute"))
    if impute_strategy != "None":
        with st.spinner("Imputing..."):
            imputer = SimpleImputer(strategy=impute_strategy.lower())
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            new_diagnostics = dataset_health_check(df_imputed, target_col)
            st.session_state[_k("df_imputed")] = df_imputed
            st.session_state[_k("diagnostics")] = new_diagnostics
            st.session_state[_k("diagnostics_target")] = target_col
            st.success(f"✅ Imputed ({impute_strategy}). Ready!")
            df = df_imputed

# -------------------------------------------------
# Proceed Controls
# -------------------------------------------------
st.divider()
st.subheader("➡️ Next Action")

col1, col2 = st.columns([2, 1])
with col1:
    if is_regression_suitable == 1 and target_type == "regression":
        if st.button("▶️ Proceed with Training", key=_k("proceed_btn"), type="primary"):
            st.session_state[_k("proceed")] = True
            for state_key in ["run_single", "run_cls", "run_comparison", "run_cls_comparison"]:
                st.session_state[_k(state_key)] = False
    else:
        st.warning("⚠️ Not suitable for regression (or classification unavailable). Clean data or override type.")
    st.session_state[_k("proceed")] = st.session_state.get(_k("proceed"), False)

with col2:
    if st.button("🔄 Refresh", key=_k("refresh")):
        for k in [key for key in st.session_state if key.startswith(PAGE_PREFIX)]:
            del st.session_state[k]
        st.rerun()

if not st.session_state[_k("proceed")]:
    st.stop()

# -------------------------------------------------
# Split & Mode (Regression Only)
# -------------------------------------------------
st.divider()
st.subheader("🔀 Data Split")
split_type = st.radio("Split Method", ["Random shuffle split", "Time-ordered split"], key=_k("split"))

st.subheader("🧠 Testing Mode")
mode = st.radio("Mode", ["Single Model Testing", "Multiple Model Comparison"], key=_k("mode"))

# -------------------------------------------------
# Regression Pipeline
# -------------------------------------------------
if target_type == "regression" and is_regression_suitable == 1:
    st.info(f"📊 **Regression Activated** for `{target_col}`")

    if mode == "Single Model Testing":
        model_type = st.selectbox("Model", ["linear", "poly", "rbf", "rf", "mlp", "xgb"], key=_k("model"))
        if st.button("🚀 Run Test", key=_k("run_single_btn")):
            st.session_state[_k("run_single")] = True

        if st.session_state.get(_k("run_single")):
            with st.spinner(f"Training {model_type}..."):
                try:
                    if model_type in ["rf", "mlp", "xgb"]:
                        model, data, metrics, _ = train_benchmark_model(df, target_col, model_type, split_type=split_type)
                    else:
                        model, data, metrics, _ = train_model(df, target_col, model_type, split_type=split_type)
                    
                    st.success("✅ Completed!")
                    st.header("📊 Metrics")
                    st.table(pd.DataFrame([metrics]))

                    y_pred = model.predict(data["X_test_scaled"])
                    preview = pd.DataFrame({"Actual": data["y_test"], "Predicted": y_pred})
                    st.subheader("🔍 Preview")
                    st.dataframe(preview.head(20))
                    st.line_chart(preview)
                    st.line_chart(pd.DataFrame({"Residuals": data["y_test"] - y_pred}))
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
                    st.exception(e)

    else:
        if st.button("🚀 Run Comparison", key=_k("run_comparison_btn")):
            st.session_state[_k("run_comparison")] = True

        if st.session_state.get(_k("run_comparison")):
            model_types = ["linear", "poly", "rbf"]
            results = []
            with st.spinner("Comparing..."):
                try:
                    base_model, data, _, _ = train_model(df, target_col, "linear", split_type=split_type)
                    X_test, y_test = data["X_test_scaled"], data["y_test"]

                    for m in model_types:
                        model, _, metrics, _ = train_model(df, target_col, m, split_type=split_type)
                        y_pred = model.predict(X_test)
                        results.append({"Model": m, **metrics, "y_pred": y_pred})

                    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
                    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
                    st.success("✅ All trained!")
                    st.header("🏆 Comparison")
                    st.table(results_df[["Rank", "Model", "MSE", "MAE", "R2"]])

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(y_test, y_test, "--", color="black", label="Ideal")
                    colors = ['red', 'green', 'purple']
                    for i, (_, r) in enumerate(results_df.iterrows()):
                        ax.scatter(y_test, r["y_pred"], alpha=0.6, label=r["Model"], c=colors[i])
                    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
                    ax.legend(); ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
                    st.exception(e)

else:
    st.warning("**Note:** Only regression supported. Use override for classification if needed.")
    st.stop()