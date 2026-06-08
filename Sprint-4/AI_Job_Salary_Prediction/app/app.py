"""
app.py  –  AI Job Salary Predictor
Streamlit frontend for Sprint 4 deployment.
"""

import os
import sys
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import load_model, predict, load_prediction_logs
from preprocessing import CATEGORICAL_COLS, NUMERICAL_COLS

MODEL_PATH = ROOT / "models" / "model.pkl"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.25rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }
        .salary-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 20px rgba(102,126,234,0.4);
        }
        .salary-label {font-size: 1rem; opacity: 0.85; margin-bottom: 0.5rem;}
        .salary-value {font-size: 3rem; font-weight: 800; letter-spacing: -1px;}
        .metric-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Option maps ────────────────────────────────────────────────────────────────
EXPERIENCE_MAP = {
    "Entry-level (EN)": "EN",
    "Mid-level (MI)": "MI",
    "Senior-level (SE)": "SE",
    "Executive (EX)": "EX",
}

EMPLOYMENT_MAP = {
    "Full-Time (FT)": "FT",
    "Part-Time (PT)": "PT",
    "Contract (CT)": "CT",
    "Freelance (FL)": "FL",
}

COMPANY_SIZE_MAP = {
    "Small (<50 employees)": "S",
    "Medium (50–250 employees)": "M",
    "Large (>250 employees)": "L",
}

JOB_TITLES = [
    "Data Scientist", "Data Engineer", "ML Engineer", "AI Engineer",
    "Data Analyst", "Research Scientist", "MLOps Engineer", "NLP Engineer",
    "Computer Vision Engineer", "Business Intelligence Analyst",
    "Data Architect", "AI Researcher", "Applied Scientist",
    "Analytics Engineer", "LLM Engineer",
]

INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Education", "Retail",
    "Manufacturing", "Media", "Consulting", "Government", "Energy",
]

EDUCATION_OPTIONS = [
    "High School", "Associate's", "Bachelor's", "Master's", "PhD",
]

COUNTRIES = [
    "US", "GB", "CA", "DE", "FR", "AU", "IN", "ES", "NL",
    "SG", "BR", "JP", "PT", "MX", "PK", "IT", "PL", "CH", "SE",
]


# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_model(str(MODEL_PATH))


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("## AI Salary Predictor")
    st.markdown("_Sprint 4 – Deployment & MLOps_")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🎯 Predict Salary", "📊 Monitoring", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Model: Gradient Boosting Regressor")
    st.caption("Dataset: AI Job Market 2024")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – PREDICT SALARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Predict Salary":
    st.markdown('<p class="main-header">💼 AI Job Salary Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Fill in the details below to estimate your market salary in USD</p>',
        unsafe_allow_html=True,
    )

    pipeline = get_model()
    if pipeline is None:
        st.error(
            "⚠️ Model not found. Please run `python src/train.py` first to train and save the model."
        )
        st.stop()

    # ── Input form ──────────────────────────────────────────────────────────
    with st.form("salary_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Role Details")
            job_title = st.selectbox("Job Title", JOB_TITLES, index=0)
            experience_level = st.selectbox(
                "Experience Level", list(EXPERIENCE_MAP.keys()), index=2
            )
            employment_type = st.selectbox(
                "Employment Type", list(EMPLOYMENT_MAP.keys()), index=0
            )
            industry = st.selectbox("Industry", INDUSTRIES, index=0)
            education_required = st.selectbox("Education Required", EDUCATION_OPTIONS, index=2)

        with col2:
            st.subheader("🏢 Company Details")
            company_size = st.selectbox(
                "Company Size", list(COMPANY_SIZE_MAP.keys()), index=1
            )
            company_location = st.selectbox("Company Location", COUNTRIES, index=0)
            employee_residence = st.selectbox("Employee Residence", COUNTRIES, index=0)
            years_experience = st.slider("Years of Experience", 0, 20, 5)

        col3, col4 = st.columns(2)
        with col3:
            remote_ratio = st.select_slider(
                "Remote Work Ratio (%)", options=[0, 50, 100], value=100,
                format_func=lambda x: {0: "On-site (0%)", 50: "Hybrid (50%)", 100: "Fully Remote (100%)"}[x],
            )
        with col4:
            benefits_score = st.slider("Benefits Score", 1.0, 5.0, 3.5, step=0.5)

        submitted = st.form_submit_button("🔮 Predict Salary")

    # ── Run prediction ───────────────────────────────────────────────────────
    if submitted:
        input_data = {
            "job_title": job_title,
            "experience_level": EXPERIENCE_MAP[experience_level],
            "employment_type": EMPLOYMENT_MAP[employment_type],
            "company_location": company_location,
            "employee_residence": employee_residence,
            "company_size": COMPANY_SIZE_MAP[company_size],
            "industry": industry,
            "education_required": education_required,
            "remote_ratio": remote_ratio,
            "years_experience": years_experience,
            "job_description_length": 800,
            "benefits_score": benefits_score,
        }

        with st.spinner("Calculating salary estimate …"):
            result = predict(input_data, pipeline=pipeline)

        salary = result["predicted_salary"]
        monthly = salary / 12
        weekly = salary / 52

        st.divider()

        # ── Result display ───────────────────────────────────────────────
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            st.markdown(
                f"""
                <div class="salary-box">
                    <div class="salary-label">Estimated Annual Salary (USD)</div>
                    <div class="salary-value">${salary:,.0f}</div>
                    <div style="opacity:0.7; margin-top:0.5rem">{job_title} · {experience_level}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Monthly", f"${monthly:,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Weekly", f"${weekly:,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Salary range gauge ───────────────────────────────────────────
        st.divider()
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=salary,
                number={"prefix": "$", "valueformat": ",.0f"},
                delta={"reference": 120000, "valueformat": ",.0f"},
                title={"text": "Salary vs Benchmark ($120K)"},
                gauge={
                    "axis": {"range": [0, 300000], "tickformat": "$,.0f"},
                    "bar": {"color": "#667eea"},
                    "steps": [
                        {"range": [0, 80000], "color": "#fef3c7"},
                        {"range": [80000, 150000], "color": "#d1fae5"},
                        {"range": [150000, 300000], "color": "#ede9fe"},
                    ],
                    "threshold": {
                        "line": {"color": "#ef4444", "width": 3},
                        "thickness": 0.75,
                        "value": 120000,
                    },
                },
            )
        )
        fig.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        # ── Input summary table ──────────────────────────────────────────
        with st.expander("📋 Input Summary"):
            st.dataframe(
                pd.DataFrame(list(input_data.items()), columns=["Feature", "Value"]),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Monitoring":
    st.markdown('<p class="main-header">📊 Model Monitoring</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Track prediction history and monitor model performance over time</p>',
        unsafe_allow_html=True,
    )

    logs = load_prediction_logs()

    if logs.empty:
        st.info("No predictions logged yet. Make a prediction first!")
        st.stop()

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Predictions", len(logs))
    m2.metric("Avg Predicted Salary", f"${logs['predicted_salary'].mean():,.0f}")
    m3.metric("Min Predicted Salary", f"${logs['predicted_salary'].min():,.0f}")
    m4.metric("Max Predicted Salary", f"${logs['predicted_salary'].max():,.0f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction History")
        fig1 = px.line(
            logs.reset_index(),
            x="timestamp",
            y="predicted_salary",
            markers=True,
            color_discrete_sequence=["#667eea"],
            labels={"predicted_salary": "Salary (USD)", "timestamp": "Time"},
        )
        fig1.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Salary Distribution")
        fig2 = px.histogram(
            logs,
            x="predicted_salary",
            nbins=20,
            color_discrete_sequence=["#764ba2"],
            labels={"predicted_salary": "Salary (USD)"},
        )
        fig2.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    if "job_title" in logs.columns:
        st.subheader("Average Salary by Job Title")
        avg_by_title = (
            logs.groupby("job_title")["predicted_salary"]
            .mean()
            .sort_values(ascending=True)
            .reset_index()
        )
        fig3 = px.bar(
            avg_by_title,
            x="predicted_salary",
            y="job_title",
            orientation="h",
            color="predicted_salary",
            color_continuous_scale="Purples",
            labels={"predicted_salary": "Avg Salary (USD)", "job_title": ""},
        )
        fig3.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Raw Prediction Log")
    st.dataframe(logs.sort_values("timestamp", ascending=False), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

    st.markdown(
        """
        ## 🚀 Sprint 4 – Deployment & MLOps

        This application is the final deployment phase of the **AI Job Salary Prediction** project.

        ### Project Pipeline
        | Step | Description | Status |
        |------|-------------|--------|
        | Data Collection | AI Job Market dataset | ✅ |
        | EDA & Preprocessing | Cleaning, feature engineering | ✅ |
        | Model Training | Gradient Boosting Regressor | ✅ |
        | ML Pipeline | Sklearn Pipeline (prep + model) | ✅ |
        | Frontend | Streamlit web app | ✅ |
        | Local Deployment | `streamlit run app.py` | ✅ |
        | Experiment Tracking | MLflow | ✅ |
        | Prediction Logging | CSV-based monitoring | ✅ |

        ### Model Information
        - **Algorithm**: Gradient Boosting Regressor
        - **Features**: Job title, experience, employment type, location, company size, work setting, remote ratio
        - **Target**: Annual salary in USD

        ### MLOps Practices Implemented
        - 📦 **ML Pipeline** – Preprocessing + encoding + model in one `sklearn.Pipeline`
        - 🔬 **Experiment Tracking** – MLflow logs parameters, metrics, and artifacts
        - 📝 **Prediction Logging** – Each inference is logged to `logs/predictions.csv`
        - 📊 **Model Monitoring** – Monitoring dashboard tracks drift over time
        - 🗂️ **Version Control** – Structured project layout for Git

        ### Tech Stack
        `scikit-learn` · `MLflow` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`
        """
    )
