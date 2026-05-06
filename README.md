# ChurnSense: Predicting Customer Churn Using Behavioral Data Analytics

![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

---

## 📌 Overview

Customer churn is a major challenge for companies because losing customers directly impacts revenue and long-term business growth. Many organizations struggle to identify customers who are likely to leave due to limited understanding of customer behavior and service usage patterns.

**ChurnSense** is a Machine Learning-based customer churn prediction system that analyzes behavioral and subscription data to identify customers who are at risk of leaving a service. The project helps businesses take proactive retention measures and supports **Sustainable Development Goal (SDG) 8 – Decent Work and Economic Growth** by improving customer retention and business sustainability.

The project uses the **IBM Telco Customer Churn Dataset**, which contains customer demographic details, subscribed services, billing information, account details, and churn status.

---

## 🎯 Objectives

- Analyze customer behavior and service usage patterns
- Predict customer churn using Machine Learning models
- Identify key factors influencing churn
- Provide interactive visualizations through a Streamlit dashboard
- Support business decision-making using predictive analytics

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Pandas & NumPy** – Data preprocessing and analysis
- **Scikit-learn** – Machine Learning models
- **Matplotlib & Seaborn** – Data visualization
- **Streamlit** – Interactive web dashboard

---

## 📂 Repository Structure

```text
ChurnSense/
│
├── data/
│   ├── raw/                    # Original IBM Telco Churn dataset
│   └── processed/              # Cleaned and transformed datasets
│
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory Data Analysis
│
├── src/
│   ├── data_pipeline.py        # Data cleaning and preprocessing
│   ├── model.py                # Model training and evaluation
│   └── inference.py            # Prediction functions for new data
│
├── app/
│   └── main.py                 # Streamlit dashboard application
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <repository-link>
cd ChurnSense
```

### 2️⃣ Install Dependencies

Ensure Python 3.9 or above is installed, then run:

```bash
pip install -r requirements.txt
```

### 3️⃣ Acquire the Dataset

Download the IBM Telco Customer Churn Dataset and place it inside:

```text
data/raw/
```

---

## 🚀 Model Training

Run the following command to preprocess the data and train the Machine Learning model:

```bash
python src/model.py
```

The trained model will be saved for future inference and dashboard integration.

---

## 📊 Launch the Streamlit Dashboard

To start the interactive dashboard:

```bash
streamlit run app/main.py
```

The dashboard enables users to:

- Explore churn statistics
- Visualize customer behavior trends
- Predict churn probability for customers
- Analyze important churn factors

---

## 📈 Features

- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Machine Learning-based churn prediction
- Model evaluation and performance metrics
- Interactive Streamlit visualization dashboard

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Exploratory Data Analysis
5. Model Training
6. Model Evaluation
7. Churn Prediction
8. Dashboard Visualization

---

## 🌍 SDG Alignment

This project aligns with:

- **SDG 8 – Decent Work and Economic Growth**

By helping companies reduce customer loss and improve retention strategies, the project contributes to sustainable business growth and economic stability.

---

## 📌 Future Enhancements

- Deploy the application on cloud platforms
- Add Deep Learning models for improved prediction accuracy
- Implement real-time churn monitoring
- Add customer segmentation and recommendation systems
- Integrate explainable AI techniques for model transparency

---

## 👨‍💻 Author

**Parvathy Porur**  
B.Tech CSE (AIML)

---

## 📜 License

This project is intended for educational and research purposes.
