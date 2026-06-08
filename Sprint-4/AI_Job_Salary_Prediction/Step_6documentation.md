# Project Documentation — AI Job Salary Prediction

---

## Problem Statement

A recruitment analytics company wants to improve how it handles hiring decisions and salary benchmarking. Right now, most salary decisions are made based on gut feeling or rough industry averages, which leads to two common problems — either the company over-offers and burns budget, or it under-offers and loses good candidates to competitors.

The specific challenges they're dealing with are:

- Not knowing what salary range to offer for a new role
- No clear understanding of how things like experience level, education, industry, or company size actually shift the salary number
- Difficulty identifying which job roles consistently pay more across different regions

So the core question this project tries to answer is:

**Can we build a model that accurately predicts the salary of a job posting, using information about the job, the company, and the candidate?**

The dataset used has 15,000 job records from the AI/ML industry, covering roles like Data Scientist, ML Engineer, AI Researcher, and more — across different countries, company sizes, and experience levels.

---

## Approach

The project was split into four sprints. Here is how each one was handled:

**Sprint 1 — Data Understanding**  
The raw dataset was loaded and explored to understand what columns were available, what the data types looked like, and where the missing values were. This step helped identify that `salary_usd` was the target and that columns like `job_title`, `experience_level`, `industry`, and `years_experience` would be the most useful predictors.

**Sprint 2 — Data Cleaning and EDA**  
Duplicate rows were removed, missing values were filled using mode (for categories) and median (for numbers), and extreme salary outliers were clipped using the 1st and 99th percentile. Exploratory charts showed that senior-level roles in the US and large companies consistently paid higher, while entry-level remote roles had wider salary variance.

**Sprint 3 — Model Building**  
Three models were tested: Ridge Regression, Random Forest, and Gradient Boosting. Gradient Boosting gave the best test R² and was selected as the final model. The full pipeline combined cleaning, one-hot encoding, standard scaling, and the model into a single `sklearn.Pipeline` object so the same steps apply consistently during both training and prediction.

**Sprint 4 — Deployment and MLOps**  
The trained pipeline was saved as `model.pkl`. A Streamlit app was built so anyone can enter job details and get a salary estimate without touching any code. MLflow was set up with a SQLite backend to log every training run — including the parameters used, the metrics produced, and the model artifact. Every prediction made through the app is also logged to a CSV file so performance can be monitored over time.

---

## Results

The final model is a **Gradient Boosting Regressor** trained on 80% of the data and tested on the remaining 20%.

| Metric | Train | Test |
|---|---|---|
| MAE (Mean Absolute Error) | $12,618 | $14,647 |
| RMSE (Root Mean Squared Error) | $17,132 | $20,332 |
| R² Score | 0.9191 | 0.8866 |
| 5-Fold CV R² | — | 0.8838 ± 0.0028 |

**Key findings from the data:**

- Experience level turned out to be the strongest predictor — senior and executive roles paid significantly more than entry-level ones regardless of location
- Company size mattered too — large companies offered on average 20–30% more than small ones for the same role
- US-based companies had the highest salary range overall, followed by GB and CA
- Remote ratio alone did not have a strong effect on salary, but fully remote roles in senior positions did trend slightly higher
- Fields like AI Research and ML Engineering consistently ranked at the top of the salary distribution

---

## How to Run the Project

**Step 1 — Clone the repository and set up the environment**

```bash
git clone https://github.com/your-username/AI_Job_Salary_Prediction.git
cd AI_Job_Salary_Prediction
pip install -r requirements.txt
```

**Step 2 — Place the data files**

Copy your CSV files into the `data/` folder:
```
data/ai_job_dataset.csv
data/cleaned_ai_job_dataset.csv
```

**Step 3 — Train the model**

```bash
python src/train.py
```

This will clean the data, train the Gradient Boosting pipeline, save `models/model.pkl`, and log everything to `mlflow.db`. You can also choose a different model:

```bash
python src/train.py --model random_forest
python src/train.py --model ridge
```

**Step 4 — Launch the web app**

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`. Fill in the job details on the Predict page and click the button to get an estimated salary.

**Step 5 — View experiment tracking**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` to compare training runs, check metrics, and download model artifacts.

**Step 6 — Test prediction from the command line**

```bash
python src/predict.py
```

This runs a sample prediction using hardcoded inputs and prints the result to the terminal. Predictions are also saved automatically to `logs/predictions.csv`.

---

*Built as part of a 4-sprint ML project covering data analysis, model development, deployment, and MLOps practices.*
