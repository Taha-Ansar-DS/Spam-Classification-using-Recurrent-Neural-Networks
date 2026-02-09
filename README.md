# Spam-Classification-using-Recurrent-Neural-Networks
Developed an NLP-based spam classification system using RNN models to automatically detect spam messages. Implemented data preprocessing, tokenization, and model evaluation using precision, recall, and F1-score. Demonstrated end-to-end machine learning pipeline development with real-world text data.

## 🚀 Overview
This project focuses on developing an intelligent spam detection system using Natural Language Processing (NLP) and deep learning techniques to automatically classify text messages as spam or legitimate. It involves data preprocessing, text feature extraction, and training a Recurrent Neural Network (RNN) model to learn sequential patterns in textual data and improve message filtering accuracy. 
This project aims to automatically identify spam messages by applying machine learning and NLP techniques, including text preprocessing, feature extraction, and sequence modeling to accurately classify messages as spam or legitimate.
The goal is to build a machine learning model that can automatically classifies messages as spam or legitimate by learning sequential text patterns and implementing a complete machine learning pipeline from preprocessing to model evaluation.

Key Features:
- Data preprocessing and cleaning
- Feature engineering
- Model training and evaluation
- Performance analysis

---

## 🧠 Problem Statement
With the rapid growth of digital communication, users are increasingly exposed to unwanted spam messages that can lead to misinformation, fraud, and reduced productivity. Manually filtering such messages is inefficient and impractical due to the large volume of data generated daily. Therefore, there is a need for an automated system that can accurately identify and filter spam messages from legitimate communication using machine learning and Natural Language Processing (NLP) techniques.
---

## 📂 Dataset
-Source: Public SMS Spam Collection dataset available on Kaggle and originally from the UCI Machine Learning Repository.
-Number of Records: 5,572 SMS messages
-Features:
 --Message: Raw text data used for Natural Language Processing and model training.
-Target Variable:
 --Category: Binary classification label indicating spam or legitimate (ham) messages.

The dataset is highly suitable for NLP-based classification tasks and demonstrates real-world challenges such as unstructured text, class imbalance, and linguistic variability.

---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## ⚙️ Project Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Result Analysis

---

## 📈 Model Performance

The following models were trained and evaluated on the SMS Spam dataset using standard classification metrics.

| Model | Accuracy | F1 Score |
|------|----------|----------|
| Logistic Regression | 96% | 0.96 |
| Random Forest | 97% | 0.97 |
| RNN (Deep Learning Model) | 96% | 0.96 |

**Evaluation Metrics Used:**
- Accuracy
- Precision
- Recall
- F1 Score

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
git clone https://github.com/Taha-Ansar-DS/Spam-Classification-using-Recurrent-Neural-Networks.git

### 2️⃣ Navigate to the project directory
cd Spam-Classification-using-Recurrent-Neural-Networks

### 3️⃣ Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow jupyter

### 4️⃣ Start Jupyter Notebook
jupyter notebook

### 5️⃣ Open the project notebook
Open the .ipynb file SpamClassification-checkpoint.ipynb

### 6️⃣ Run the project
Click "Kernel" → "Restart & Run All"
OR press Shift + Enter to run each cell.


