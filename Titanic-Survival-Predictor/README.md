# 🚢 Titanic Survival Predictor

This project demonstrates a **Logistic Regression** model to predict whether a passenger survived the Titanic disaster. It uses passenger data such as age, gender, class, and fare, and includes data visualization, preprocessing, model training, and evaluation.

---

## 📊 Problem Statement

Given passenger data, can we predict whether a passenger survived the Titanic sinking?  
This project explores patterns in the data and builds a classification model to solve this problem.

---

## 🛠️ Technologies Used

- Python 🐍  
- Pandas 📊  
- Seaborn & Matplotlib 📈  
- scikit-learn 🤖  

---

## 📁 Project Structure

Titanic-Survival-Predictor/
├── TitanicPredictor.py         # Main script
├── Titanic-Dataset.csv         # Dataset file
└── README.md                   # Project documentation

---

## 📦 Dataset (Titanic-Dataset.csv)

| Column      | Description                                                          |
| ----------- | -------------------------------------------------------------------- |
| PassengerId | Unique ID for each passenger                                         |
| Survived    | Survival (0 = No, 1 = Yes)                                           |
| Pclass      | Ticket class (1st, 2nd, 3rd)                                         |
| Name        | Passenger’s full name                                                |
| Sex         | Gender                                                               |
| Age         | Age in years                                                         |
| SibSp       | Number of siblings/spouses aboard                                    |
| Parch       | Number of parents/children aboard                                    |
| Ticket      | Ticket number                                                        |
| Fare        | Ticket fare                                                          |
| Cabin       | Cabin number                                                         |
| Embarked    | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 📈 Visualizations
The script generates visual insights such as:

- Survival distribution
- Survival by gender, class, age, and fare
- Histograms for Age and Fare
- Countplots grouped by category

---

## ⚙️ Preprocessing Steps
Missing value treatment (mean for Age, mode for Embarked)

Dropped irrelevant columns (Name, Ticket, Cabin, etc.)

Label encoding for categorical features (Sex, Embarked)

Feature scaling with StandardScaler

---

## 🚀 How to Run
### ✅ Prerequisites
Install the required Python libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

▶️ Run the Script

```bash
python TitanicPredictor.py
```
---

📤 Sample Output
```bah
Titanic Survival Predictor
First 5 rows of the dataset:

Survived and Non-survived passengers

Accuracy using Logistic Regression: 79.10%

Confusion matrix:
[[126  23]
 [ 28  91]]

Classification report:
              precision    recall  f1-score   support
         0       0.82      0.85      0.84       149
         1       0.80      0.76      0.78       119
```

--- 

## 📌 Notes
The model is a simple logistic regression, ideal for binary classification.

You can test accuracy and behavior by tweaking features or preprocessing logic.

---

## 🙌 Acknowledgements
Dataset adapted from Kaggle Titanic Challenge

Built using Python, Pandas, Seaborn, Matplotlib, and scikit-learn
