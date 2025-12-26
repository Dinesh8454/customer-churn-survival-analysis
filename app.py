import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

st.title("Customer Churn Survival Analysis")

# Load data
df = pd.read_excel("churn_data.xlsx")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocessing
df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)

# Kaplan-Meier
st.subheader("Kaplan–Meier Survival Curve")
kmf = KaplanMeierFitter()
kmf.fit(df['tenure'], df['churn'])

fig, ax = plt.subplots()
kmf.plot(ax=ax)
ax.set_xlabel("Tenure (Months)")
ax.set_ylabel("Survival Probability")
st.pyplot(fig)

# Cox Model
st.subheader("Cox Proportional Hazards Model")
cph = CoxPHFitter()
cph.fit(df.drop(columns=['customer_id']), duration_col='tenure', event_col='churn')
st.text(cph.summary)
