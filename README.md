# 🤖 Machine Learning Projects — Innomatics Research Labs

A end-to-end ML internship portfolio covering the full data science lifecycle — from exploratory analysis and classical ML through to production deployment with MLOps practices. The repo is organized into four sprints, each building on the last.

---

## 📁 Repository Structure

```
Machine-Learning-Projects-Innomatics/
│
├── Sprint-1/                          # EDA & Data Preprocessing
├── Sprint-2/                          # Machine Learning Modelling
├── Sprint -3/                         # Advanced ML & Feature Engineering
├── Sprint-4/
│   └── AI_Job_Salary_Prediction/      # Deployment & MLOps (Full App)
│       ├── app/
│       │   └── app.py                 # Streamlit frontend
│       ├── src/
│       │   ├── preprocessing.py       # Custom sklearn transformer
│       │   ├── predict.py             # Inference + logging
│       │   └── train.py               # Model training script
│       ├── models/
│       │   └── model.pkl              # Trained pipeline (Git LFS)
│       ├── data/
│       │   └── cleaned_ai_job_dataset.csv
│       └── logs/
│           └── predictions.csv        # Live prediction log
│
├── requirements.txt
├── runtime.txt
└── ML_Project_Document.docx.pdf
```

---

## 🚀 Sprints Overview

### Sprint 1 — EDA & Data Preprocessing
Exploratory data analysis and data cleaning fundamentals.

- Handling missing values, outliers, and duplicates
- Univariate and bivariate analysis
- Feature distributions and correlation heatmaps
- Data visualization with Matplotlib and Seaborn

**Tools:** `pandas` · `numpy` · `matplotlib` · `seaborn` · `Jupyter Notebook`

---

### Sprint 2 — Machine Learning Modelling
Core supervised learning algorithms applied to structured datasets.

- Regression: Linear, Ridge, Lasso, Decision Tree, Random Forest
- Classification: Logistic Regression, KNN, SVM, Naive Bayes
- Model evaluation: cross-validation, confusion matrix, ROC-AUC
- Hyperparameter tuning with GridSearchCV

**Tools:** `scikit-learn` · `pandas` · `numpy` · `Jupyter Notebook`

---

### Sprint 3 — Advanced ML & Feature Engineering
Advanced techniques for improving model performance.

- Feature engineering and selection
- Ensemble methods: Gradient Boosting, XGBoost, AdaBoost
- Pipeline construction with `sklearn.Pipeline`
- Experiment tracking with MLflow

**Tools:** `scikit-learn` · `xgboost` · `mlflow` · `Jupyter Notebook`

---

### Sprint 4 — AI Job Salary Prediction (Deployment & MLOps) 🏆

A fully deployed ML web application that predicts annual AI job salaries in USD based on role, experience, location, and company attributes.

#### Live Demo
> Deployed on [Streamlit Cloud](https://streamlit.io/cloud)

#### Features
- 🔮 **Salary Prediction** — Real-time predictions using a trained Gradient Boosting pipeline
- 📊 **Monitoring Dashboard** — Tracks prediction history, salary distribution, and per-job-title averages
- 🔁 **Auto-Retrain on Deploy** — If the saved model is incompatible (Python version mismatch), the app retrains automatically from the CSV data at startup
- 📝 **Prediction Logging** — Every inference is saved to `logs/predictions.csv`

#### Model Details
| Property | Value |
|----------|-------|
| Algorithm | Gradient Boosting Regressor |
| n_estimators | 300 |
| learning_rate | 0.05 |
| max_depth | 5 |
| Target | Annual salary in USD |
| Storage | Git LFS (model.pkl > 100 MB) |

#### MLOps Practices
- **Sklearn Pipeline** — preprocessing + encoding + model as a single serialisable object
- **MLflow** — experiment tracking (parameters, metrics, artifacts)
- **Prediction Logging** — CSV-based monitoring of live inference
- **Streamlit Cloud** — one-click CI/CD deployment from GitHub
- **Git LFS** — large model files tracked without bloating the repo

#### Project Structure (Sprint 4)
```
AI_Job_Salary_Prediction/
├── app/app.py            # Streamlit UI
├── src/
│   ├── preprocessing.py  # AIJobPreprocessor (custom transformer)
│   ├── predict.py        # load_model(), predict(), load_prediction_logs()
│   └── train.py          # Training script
├── models/model.pkl      # Serialised pipeline (Git LFS)
├── data/                 # Raw & cleaned datasets
└── logs/predictions.csv  # Inference log
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+ (3.11 recommended for local; Streamlit Cloud uses 3.14)
- Git with [Git LFS](https://git-lfs.github.com/) installed

### Clone & Install

```bash
# Clone the repo (LFS files download automatically)
git clone https://github.com/Siva-pa/Machine-Learning-Projects-Innomatics.git
cd Machine-Learning-Projects-Innomatics

# Install dependencies
pip install -r requirements.txt
```

### Run the App Locally

```bash
cd Sprint-4/AI_Job_Salary_Prediction
streamlit run app/app.py
```

---

## 📦 Dependencies

Key packages (see `requirements.txt` for full list):

```
streamlit
scikit-learn
pandas
numpy
plotly
joblib
mlflow
```

---

## 🗂️ Git LFS

Model files (`.pkl`) exceed GitHub's 100 MB limit and are stored with Git LFS.

```bash
# First-time setup
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

If you pull the repo and the model file is a pointer (not the actual file), run:

```bash
git lfs pull
```

---

## 📄 Documentation

Full project documentation is available in [`ML_Project_Document.docx.pdf`](./ML_Project_Document.docx.pdf).

---

## 👤 Author

**Siva** — Innomatics Research Labs ML Internship  
GitHub: [@Siva-pa](https://github.com/Siva-pa)

---

## 📜 License

This project is for educational purposes as part of the Innomatics Research Labs internship programme.
