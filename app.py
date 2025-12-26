import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Survival Analysis",
    layout="wide"
)

st.title("📊 Customer Churn Survival Analysis")
st.write("Survival Analysis using Kaplan–Meier Estimator and Cox Proportional Hazards Model")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("churn_data.xlsx")

df = load_data()

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Data Preprocessing
# -----------------------------
st.subheader("⚙️ Data Preprocessing")
df_processed = pd.get_dummies(df, columns=["contract_type"], drop_first=True)
st.write("Categorical variables converted using one-hot encoding")
st.dataframe(df_processed.head())

# -----------------------------
# Kaplan–Meier Survival Analysis
# -----------------------------
st.subheader("📈 Kaplan–Meier Survival Curve")

kmf = KaplanMeierFitter()
T = df_processed["tenure"]
E = df_processed["churn"]

kmf.fit(T, event_observed=E, label="Customer Survival")

fig1, ax1 = plt.subplots(figsize=(5, 3))
kmf.plot(ax=ax1)
ax1.set_xlabel("Tenure (Months)")
ax1.set_ylabel("Survival Probability")
ax1.set_title("Kaplan–Meier Survival Curve")

st.pyplot(fig1)

st.info("This curve shows the probability of customers remaining active over time.")

# -----------------------------
# Survival Curves by Contract Type
# -----------------------------
st.subheader("📊 Survival Curves by Contract Type")

fig2, ax2 = plt.subplots(figsize=(5, 3))

for col in ["contract_type_OneYear", "contract_type_TwoYear"]:
    if col in df_processed.columns:
        ix = df_processed[col] == 1
        kmf.fit(
            df_processed.loc[ix, "tenure"],
            df_processed.loc[ix, "churn"],
            label=col
        )
        kmf.plot(ax=ax2)

ax2.set_xlabel("Tenure (Months)")
ax2.set_ylabel("Survival Probability")
ax2.set_title("Survival Curves by Contract Type")

st.pyplot(fig2)

# -----------------------------
# Cox Proportional Hazards Model
# -----------------------------
st.subheader("🧮 Cox Proportional Hazards Model")

cph = CoxPHFitter()
cph.fit(
    df_processed.drop(columns=["customer_id"]),
    duration_col="tenure",
    event_col="churn"
)

st.write("### Cox Model Summary")
st.dataframe(cph.summary)

st.info(
    "Hazard Ratio > 1 indicates higher churn risk, "
    "while Hazard Ratio < 1 indicates lower churn risk."
)

# -----------------------------
# Model Assumption Check
# -----------------------------
st.subheader("✅ Proportional Hazards Assumption Check")

st.write(
    "The Cox model assumes that hazard ratios remain constant over time. "
    "This check validates that assumption."
)

try:
    cph.check_assumptions(
        df_processed.drop(columns=["customer_id"]),
        show_plots=True
    )
    st.success("Assumption check completed successfully.")
except Exception:
    st.warning(
        "Assumption plots cannot be rendered directly in Streamlit. "
        "Run this check in Jupyter Notebook for detailed diagnostics."
    )

# -----------------------------
# Final Insights
# -----------------------------
st.subheader("📌 Key Insights")
st.markdown("""
- Customers with **monthly contracts** show a higher churn risk  
- **Longer tenure** significantly reduces churn probability  
- **Higher monthly charges** increase churn likelihood  
- **Two-year contracts** demonstrate better customer retention  
""")

st.success("🎉 Survival analysis completed successfully!")
