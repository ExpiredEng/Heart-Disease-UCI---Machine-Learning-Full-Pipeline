
# ❤️ Heart Disease Prediction – Machine Learning Pipeline

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/ExpiredEng/heart-disease-prediction)](https://github.com/ExpiredEng/heart-disease-prediction/issues)

---

## 📌 Project Overview

This project implements a **complete machine learning pipeline** on the **UCI Heart Disease dataset**, aimed at analyzing, predicting, and visualizing heart disease risks.

It combines **supervised and unsupervised learning** approaches and provides a **Streamlit web app** for real-time predictions. The app can also be deployed publicly using **Ngrok**.

---

## 🎯 Objectives

* Data Preprocessing & Cleaning (missing values, encoding, scaling)
* Feature Selection (RFE, Chi-Square, Feature Importance)
* Dimensionality Reduction (PCA)
* Supervised Learning Models: Logistic Regression, Decision Tree, Random Forest, SVM
* Unsupervised Learning: K-Means Clustering, Hierarchical Clustering
* Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV)
* Model Export (`final_model.pkl`)
* Streamlit Web UI for real-time predictions
* Deployment via Ngrok
* Full documentation and reproducible GitHub repository

---

## 🗂️ Repository Structure

```
heart-disease-prediction/
│
├─ data/                     # Raw and processed datasets
│   ├─ heart_disease.csv
│   ├─ heart_disease_cleaned.csv
│   └─ heart_disease_pca.csv
│
├─ notebooks/                # Jupyter Notebooks
│   ├─ 01_data_preprocessing.ipynb
│   ├─ 02_feature_selection.ipynb
│   ├─ 03_pca_dimensionality_reduction.ipynb
│   ├─ 04_supervised_learning.ipynb
│   └─ 05_unsupervised_learning.ipynb
│
├─ models/                   # Trained and saved models
│   └─ final_model.pkl
│
├─ results/                  # Results, figures, clustering outputs
│   └─ clustering_results.csv
│
├─ ui/                       # Streamlit web app
│   └─ app.py
│
├─ requirements.txt           # Python dependencies
└─ README.md                  # Project documentation
```

---

## 🛠️ Tools & Libraries

* **Python 3.11+**
* **Data Analysis & ML:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
* **Model Export:** `joblib`
* **Web App:** `streamlit`
* **Optional Deployment:** `ngrok`

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ExpiredEng/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
cd ui
streamlit run app.py
```

Open your browser at 👉 [http://localhost:8501](http://localhost:8501)

### 4️⃣ (Optional) Deploy via Ngrok

```bash
ngrok http 8501
```

Share the **generated public link** for online access.

---

## 📊 Results

* Cleaned dataset with selected features
* PCA-transformed dataset & variance plots
* Trained supervised and unsupervised models
* Performance metrics: Accuracy, Precision, Recall, F1, ROC AUC
* Optimized model (`final_model.pkl`)
* Interactive Streamlit web app for predictions

---

## 🌍 Dataset

* **Source:** UCI Heart Disease Database
* [UCI Repository – Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

---

## ✨ Author

* 👤 **Elsayed Ashraf Bakry**
* 📧 [sayedworkacc@gmail.com](mailto:sayedworkacc@gmail.com)

---

Do you want me to add that?
