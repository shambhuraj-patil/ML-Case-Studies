# 💳 Fraud Detection using Machine Learning

This repository contains a complete end-to-end machine learning project that detects fraudulent transactions. It covers all key stages of a typical ML pipeline — data cleaning, preprocessing, visualization, class balancing, model building, hyperparameter tuning, and evaluation.

---

## 📁 File Structure

Fraud_Detection.py # Main Python script with all logic

fraud_detection.csv # Input dataset containing transaction records

README.md # Project documentation

---

## ✅ Step 1 – Data Loading & Cleaning

- Load dataset using Pandas
- Handle missing values (mean imputation for `Amount`)
- Drop unnecessary columns like `TransactionID` and `CustomerID`
- Remove outliers using IQR for numeric columns

### Concepts:
- Data wrangling
- Missing value handling
- Outlier detection with Interquartile Range

---

## ✅ Step 2 – Exploratory Data Analysis (EDA)

- Count of fraud vs non-fraud transactions
- Fraud distribution by transaction type
- Boxplot visualizations for numeric fields

### Concepts:
- Seaborn and Matplotlib for visualization
- Distribution analysis
- Category-wise fraud breakdown

---

## ✅ Step 3 – Preprocessing & Feature Engineering

- One-hot encoding for categorical features: `TransactionType`, `Location`, `DeviceType`, `TimeOfDay`
- Feature scaling using `StandardScaler`
- SMOTE for handling class imbalance

### Concepts:
- Encoding categorical variables
- Feature scaling
- Class balancing techniques (SMOTE)

---

## ✅ Step 4 – Model Training & Evaluation

- Logistic Regression model (baseline)
- Random Forest with `GridSearchCV` for hyperparameter tuning
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score

### Concepts:
- Supervised classification
- Model training and testing split
- Hyperparameter optimization

---

## ✅ Step 5 – Final Output & Comparison

- Metrics printed for both models
- Best hyperparameters displayed for Random Forest
- Overall comparison for real-world fraud detection use cases

---

## 🛠 Technologies Used

- Python 3.x
- Libraries:  
  `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `imbalanced-learn`

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/fraud-detection-ml.git
cd fraud-detection-ml
```

# Install dependencies
```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
```

# Run the project
```bash
python Fraud_Detection.py
```

---

## 📌 Notes

- Ensure the CSV file is named fraud_detection.csv and placed in the same directory as the Python script.

- Designed for educational and demonstration purposes.

