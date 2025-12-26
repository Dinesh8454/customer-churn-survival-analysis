import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Customer Churn Survival Analysis", layout="wide")
st.title("📊 Customer Churn Survival Analysis")
st.write("Survival Analysis using Kaplan–Meier and Cox Proportional Hazards Model")

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
st.dataframe(df_processed.head())

# -----------------------------
# Kaplan–Meier Survival Curve
# -----------------------------
st.subheader("📈 Kaplan–Meier Survival Curve")

kmf = KaplanMeierFitter()
kmf.fit(df_processed["tenure"], df_processed["churn"])

# 🔥 FORCE SMALL SIZE
fig1, ax1 = plt.subplots(figsize=(3.5, 2.2), dpi=120)
kmf.plot(ax=ax1)
ax1.set_xlabel("Tenure (Months)", fontsize=8)
ax1.set_ylabel("Survival Probability", fontsize=8)
ax1.set_title("Kaplan–Meier Curve", fontsize=9)
ax1.tick_params(labelsize=7)
plt.tight_layout()

st.pyplot(fig1, use_container_width=False)

# -----------------------------
# Survival Curves by Contract Type
# -----------------------------
st.subheader("📊 Survival Curves by Contract Type")

fig2, ax2 = plt.subplots(figsize=(3.5, 2.2), dpi=120)

for col in ["contract_type_OneYear", "contract_type_TwoYear"]:
    if col in df_processed.columns:
        ix = df_processed[col] == 1
        kmf.fit(df_processed.loc[ix, "tenure"],
                df_processed.loc[ix, "churn"],
                label=col)
        kmf.plot(ax=ax2)

ax2.set_xlabel("Tenure (Months)", fontsize=8)
ax2.set_ylabel("Survival Probability", fontsize=8)
ax2.set_title("Survival by Contract", fontsize=9)
ax2.tick_params(labelsize=7)
plt.tight_layout()

st.pyplot(fig2, use_container_width=False)

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

st.dataframe(cph.summary)

# -----------------------------
# Insights
# -----------------------------
st.subheader("📌 Key Insights")
st.markdown("""
- Monthly contracts show **higher churn risk**
- Longer tenure reduces churn probability
- Higher monthly charges increase churn likelihood
- Two-year contracts retain customers better
""")

st.success("✅ Analysis completed successfully")
