# 🎯 Play Predictor - ML Case Study

This project demonstrates a simple machine learning classification model using the K-Nearest Neighbors (KNN) algorithm. It predicts whether a game will be played based on weather conditions and temperature.

## 📂 Project Structure
ML-Case-Studies/
└── Play-Predictor/
├── PlayPredictor.py
└── PlayPredictor.csv

## 🧠 Algorithm Used
- K-Nearest Neighbors (KNN) from `sklearn.neighbors`  
- Label Encoding for categorical data

## 📌 Dataset (PlayPredictor.csv)
The dataset contains the following columns:
- `Weather`: Categorical (e.g., Sunny, Rainy, Overcast)  
- `Temperature`: Categorical (e.g., Hot, Mild, Cool)  
- `Play`: Target variable (Yes/No)

### Sample:

| Weather  | Temperature | Play |
|----------|-------------|------|
| Sunny    | Hot         | No   |
| Overcast | Cool        | Yes  |
| Rainy    | Mild        | Yes  |

---

## 🚀 How It Works
1. Reads the CSV file using `pandas`.  
2. Encodes the categorical variables using `LabelEncoder`.  
3. Trains a KNN classifier using weather and temperature as features.  
4. Predicts the result for given encoded input `[0, 2]` (Overcast, Hot).

---

## ▶️ Run the Application

### ✅ Prerequisites
Ensure you have Python 3 and the required libraries installed:

```bash
pip install pandas scikit-learn
```

### ▶️ Run the Script
```bash
python PlayPredictor.py
```

📤 Sample Output
```bash
Play Predictor application using K Nearest Neighbor
Size of dataset : 14
Names of feature are ['Weather', 'Temperature']
[0 0 1 2 2 2 1 0 0 2 0 1 1 2]  # Encoded Weather
[1 1 1 0 0 0 0 0 0 0 0 0 1 0]  # Encoded Temperature
[1] # Prediction: 1 likely maps to 'Yes'
```
