# 🍷 Wine Type Classifier

This project uses **Logistic Regression** to classify different types of wines based on their chemical properties.

---

## 📁 Files

- `WineClassifier.py` — Python script implementing the classifier
- `WinePredictor.csv` — Dataset containing wine features and classes
- `README.md` — This file

---

## 🧪 Project Overview

- Loads the wine dataset (`WinePredictor.csv`)
- Visualizes features using Seaborn pairplot
- Preprocesses data with feature scaling
- Splits data into training and testing sets
- Trains a Logistic Regression model
- Evaluates the model using accuracy score
- Makes a prediction for a custom input sample

---

## 🔍 Features Used

- Alcohol
- Malic Acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

---

## 🧠 Model Used

- Logistic Regression from `scikit-learn` with `max_iter=10000`

---

## 📊 Evaluation Metrics

- Accuracy Score

---

## 📈 How to Run

1. Install the required packages (if not already installed):

```bash
pip install pandas matplotlib seaborn scikit-learn numpy
