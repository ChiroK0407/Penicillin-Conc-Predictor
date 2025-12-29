import streamlit as st
import pandas as pd

st.title("Penicillin Fermentation Prediction Dashboard")

# -----------------------------
# Introduction & Aims
# -----------------------------
st.markdown("## 🧪 Introduction: Penicillin Fermentation")
st.write("""
Penicillin fermentation is the **biotechnological process** used to produce penicillin,
one of the most important antibiotics in medical history. It is carried out using
*Penicillium chrysogenum* strains in large-scale bioreactors.

### Why is it industrially relevant?
- Penicillin remains a **critical pharmaceutical product**, with global demand in healthcare.
- Industrial fermentation represents one of the earliest and most successful examples of
  **bioprocess scale-up**, bridging microbiology and chemical engineering.
- Modern facilities use advanced **process analytical technology (PAT)** and **control strategies**
  to maximize yield, reduce variability, and ensure product quality.

### How does the process work?
1. **Microbial growth phase:** The fungus grows in a nutrient-rich medium, consuming substrates
   like glucose and nitrogen sources.
2. **Production phase:** Once growth slows, the organism diverts metabolism toward secondary
   metabolite production — penicillin.
3. **Control variables:** Aeration, agitation, pH, temperature, and substrate feeding are tightly
   regulated to optimize penicillin yield.
4. **Mechanism:** Penicillin biosynthesis involves the condensation of amino acid precursors
   (valine, cysteine) into the β-lactam ring structure, catalyzed by specific enzymes.
5. **Industrial challenge:** Maintaining dissolved oxygen, substrate balance, and precursor
   concentrations while avoiding inhibitory by-products.

This dashboard leverages **simulation data (IndPenSim)** to model and predict penicillin
concentration under varying process conditions, helping us explore modern monitoring
and control challenges in biopharmaceutical manufacturing.
""")

st.markdown("## 🎯 Our Aims")
st.write("""
This multipage application has been developed to **model, compare, and test predictive algorithms**
on industrial-scale fermentation datasets. Our primary goals are:

- To provide a **scientifically rigorous platform** for predicting penicillin fermentation outcomes.
- To enable **comparison of multiple models** (SVR, scratch implementations, Random Forest, MLP, XGBoost).
- To allow **interactive prediction** with manual inputs and error margins.
- To support **general testing** on external datasets, ensuring robustness when evaluated by peers or instructors.
- To maintain **clarity and transparency** in all visualizations, metrics, and documentation.
""")

st.markdown("## 📚 How This App Helps Achieve Our Aims")
st.write("""
- **Page 1 (Model Testing & Diagnostics):** Upload a dataset, select a model, and run auto‑tune with diagnostics to compare tuned vs untuned performance.
- **Page 2 (Model Comparison):** Benchmarks multiple models side‑by‑side with metrics, overlay plots, and scatter plots.
- **Page 3 (Interactive Prediction):** Allows manual input of features to generate predictions with ± error bounds.
- **Page 4 (General Testing):** Provides flexibility to test our models on any dataset, including those from other groups.
""")

# -----------------------------
# Index Section
# -----------------------------
st.markdown("## 📑 Model Index (Short Forms)")
index = {
    "linear": "Support Vector Regression (SVR) with Linear Kernel",
    "poly": "Support Vector Regression (SVR) with Polynomial Kernel",
    "rbf": "Support Vector Regression (SVR) with Radial Basis Function Kernel",
    "linear_scratch": "Scratch Implementation of Linear SVR",
    "poly_scratch": "Scratch Implementation of Polynomial SVR",
    "rbf_scratch": "Scratch Implementation of RBF SVR",
    "rf": "Random Forest Regressor",
    "mlp": "Multi-Layer Perceptron Regressor (Neural Network)",
    "xgb": "Extreme Gradient Boosting Regressor (XGBoost)"
}
st.table(pd.DataFrame.from_dict(index, orient="index", columns=["Full Model Name"]))

# -----------------------------
# Documentation & Citation
# -----------------------------
st.markdown("## 📖 Documentation & Citation")
st.write("""
This work builds upon the **IndPenSim industrial-scale penicillin fermentation simulation**,
a benchmark dataset for biopharmaceutical process monitoring and control.

**Citation:**
Goldrick, S., Duran-Villalobos, C.A., Jankauskas, K., Lovett, D., Farida, S.S., Lennox, B.  
*Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process.*  
Computers and Chemical Engineering, 130 (2019), 106471.  
DOI: [10.1016/j.compchemeng.2019.05.037](https://doi.org/10.1016/j.compchemeng.2019.05.037)

**Dataset Source:**  
[www.industrialpenicillinsimulation.com](http://www.industrialpenicillinsimulation.com)
""")

st.divider()
st.markdown("### 🔄 Reset Application")
if st.button("Reset All Pages"):
    for key in list(st.session_state.keys()):
        st.session_state[key] = None
    st.rerun()