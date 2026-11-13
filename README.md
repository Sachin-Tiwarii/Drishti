# 🛡️ Drishti

**AI-Powered Crime Analytics & Weekly Forecasting for Indore, Madhya Pradesh.**

Drishti is a personal machine learning project that explores how data science can help analyze crime patterns and forecast weekly incident counts across different localities.

Built using **Python, Streamlit, XGBoost, Random Forest, and K-Nearest Neighbors (KNN)**, Drishti provides an interactive dashboard for exploring area-level crime patterns, comparing machine learning models, and visualizing geographical crime distributions.

> **Disclaimer:** Drishti currently uses synthetic demonstration data, not official police records or verified crime statistics. Its predictions are intended for research and demonstration purposes only.

---

## 📊 Project Overview

- **Project Type:** Personal Machine Learning Project
- **Domain:** Crime Analytics and Forecasting
- **Location:** Indore, Madhya Pradesh, India
- **Objective:** Analyze spatial and temporal crime patterns and forecast weekly incident counts by locality.
- **Application:** Interactive Streamlit Dashboard

## 🧠 Machine Learning Models

The project uses three regression algorithms to explore and compare weekly crime forecasting performance:

- **XGBoost Regressor** – Gradient boosting for capturing complex patterns.
- **Random Forest Regressor** – Ensemble learning using multiple decision trees.
- **K-Nearest Neighbors (KNN)** – Prediction based on similarities between observations.

### Model Evaluation

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Accuracy within a ±20 prediction tolerance

Model performance metrics are based on the available demonstration dataset and do not establish real-world predictive accuracy.

## ✨ Features

- Interactive crime analytics dashboard.
- Locality-based crime data exploration.
- Weekly crime incident forecasting.
- Comparison of XGBoost, Random Forest, and KNN.
- Model performance visualization.
- Geographical crime density heatmaps.
- Area-specific weekly crime trend analysis.
- Interactive maps using PyDeck.
- Synthetic crime category distribution analysis.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Altair, PyDeck |
| Data Storage | CSV, Pickle |

## 📁 Project Structure

```text
Drishti/
│
├── dataset/
│   ├── area_reference.csv
│   ├── crime_density_by_area.csv
│   ├── grouped_week.csv
│   └── synthetic_category_weekly.csv
│
├── models/
│   ├── crime_model_results.pkl
│   ├── knn_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── app.py
├── crime_detection_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Sachin-Tiwarii/Drishti.git
cd Drishti
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal:

```text
http://localhost:8501
```

## 🔮 Future Improvements

- Integrate verified and publicly available crime datasets.
- Improve forecasting through hyperparameter optimization.
- Explore spatiotemporal deep learning models.
- Add more detailed locality-level analytics.
- Develop exportable crime analytics reports.
- Improve model evaluation using real-world historical data.

---

*Built with Python and Machine Learning to explore data-driven approaches to urban safety.*

**Note:** Indore Sentinel AI is an experimental academic/personal project. Its outputs are intended for demonstration and research purposes only, not as evidence of actual crime activity.