# ❤️ Heart Disease Prediction – Machine Learning Pipeline  

## 📌 Project Overview  
This project implements a **comprehensive machine learning pipeline** on the **UCI Heart Disease dataset**.  
The goal is to **analyze, predict, and visualize heart disease risks** using both supervised and unsupervised learning techniques.  

The project also includes a **Streamlit web app** for real-time predictions, and can be deployed online using **Ngrok**.  

---

## 🎯 Objectives  
- ✅ Data Preprocessing & Cleaning (handle missing values, encoding, scaling)  
- ✅ Dimensionality Reduction with **PCA**  
- ✅ Feature Selection (RFE, Chi-Square, Feature Importance)  
- ✅ Supervised Learning Models: Logistic Regression, Decision Tree, Random Forest, SVM  
- ✅ Unsupervised Learning: K-Means Clustering, Hierarchical Clustering  
- ✅ Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV)  
- ✅ Model Export (`final_model.pkl`)  
- ✅ Streamlit Web UI for predictions  
- ✅ Deployment via **Ngrok**  
- ✅ Full GitHub Repository with documentation  

---

## 🛠️ Tools & Libraries  
- **Python 3.11+**  
- **Libraries:**  
  - pandas, numpy, scikit-learn, matplotlib, seaborn  
  - joblib (for saving/loading models)  
  - streamlit (for the web app)  
- **Optional:** Ngrok (for public deployment)  

---

## 📂 Project Structure  
Heart_Disease_Project/
│── data/
│ └── heart_disease.csv
│
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ └── 06_hyperparameter_tuning.ipynb
│
├── models/
│ └── final_model.pkl
│
├── ui/
│ └── app.py # Streamlit UI
│
├── results/
│ └── evaluation_metrics.txt
│
├── deployment/
│ └── ngrok_setup.txt
│
├── README.md
├── requirements.txt
└── .gitignore


---

## 🚀 How to Run  

### 1️⃣ Clone the Repository  

git clone https://github.com/ExpiredEng/heart-disease-prediction.git
cd heart-disease-prediction

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run the Streamlit App
cd ui
streamlit run app.py


Go to 👉 http://localhost:8501
 in your browser.

### 4️⃣ (Optional) Deploy via Ngrok
ngrok http 8501


Share the generated public link.

📊 Results

Cleaned dataset with selected features

PCA-transformed dataset & variance plots

Trained supervised and unsupervised models

Performance metrics (Accuracy, Precision, Recall, F1, AUC)

Optimized model (final_model.pkl)

Interactive prediction web app

🌍 Dataset

The dataset comes from the UCI Heart Disease Database:
👉 UCI Repository – Heart Disease

✨ Author

👤 Elsayed Ashraf Bakry
📧 [sayedworkacc@gmail.com]


---



