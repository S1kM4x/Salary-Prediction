# app.py
import pandas as pd
from joblib import load
import streamlit as st

model_path = "../models/model_salary.joblib"
data_path  = "../data/Salary_Data.csv"

@st.cache_resource
def load_model(path: str):
    return load(path)

@st.cache_data
def load_categories(csv_path: str):
    df = pd.read_csv(csv_path, usecols=["Education Level", "Job Title"])
    edu = sorted(df["Education Level"].dropna().astype(str).unique().tolist())
    job = sorted(df["Job Title"].dropna().astype(str).unique().tolist())
    return edu, job

st.set_page_config(page_title="Salary Estimator", layout="centered")
st.title("Salary Estimator")
st.caption("Pick from dropdowns.")

pipe = load_model(model_path)
edu_opts, job_opts = load_categories(data_path)

col1, col2 = st.columns(2)
with col1:
    edu_sel = st.selectbox("Education Level", options=edu_opts, index=0)
with col2:
    job_sel = st.selectbox("Job Title", options=job_opts, index=0)

years_exp = st.number_input("Years of Experience", min_value=0, step=1, value=3)

if st.button("Predict salary"):
    row = pd.DataFrame([{
        "Education Level": edu_sel,
        "Job Title": job_sel,
        "Years of Experience": float(years_exp),
    }])
    pred = float(pipe.predict(row)[0])
    st.success(f"Estimated salary: {pred:,.0f}")
    with st.expander("Inputs used"):
        st.write(row)
