# 🧠 Head-Brain Size Predictor
This project demonstrates a simple Linear Regression model using scikit-learn to predict brain weight based on head size. It uses a dataset of human head sizes and corresponding brain weights to train the model.

## 📊 Problem Statement
The goal is to understand the relationship between head size (in cm³) and brain weight (in grams) and use a linear regression model to predict brain weight from head size.

## 🛠️ Technologies Used
- Python 🐍
- Pandas 📊
- Matplotlib 📈
- scikit-learn 🤖

---

## 📁 Project Structure
HeadBrain-Predictor/
├── HeadBrain.csv              # Dataset
├── HeadBrainSizePredictor.py  # Main script
└── README.md                  # Project documentation

---

### 📦 Dataset (HeadBrain.csv)

The dataset contains the following columns:

| Column               | Description                      |
|----------------------|----------------------------------|
| Gender               | Male or Female                   |
| Age Range            | Age group (e.g., 20-30)          |
| Head Size (cm³)      | Volume of the head in cubic cm   |
| Brain Weight (grams) | Weight of the brain in grams     |

---

#### 📌 Example Rows

| Gender | Age Range | Head Size (cm³) | Brain Weight (grams) |
|--------|-----------|------------------|------------------|
| Male   | 20-30     | 4512             | 1530             |
| Female | 30-40     | 3738             | 1290             |

---

### 🚀 How to Run
#### ✅ Prerequisites
Ensure you have Python installed with the following packages:

```bash
pip install pandas matplotlib scikit-learn
```

▶️ Run the Application

```bash
python HeadBrainSizePredictor.py
```
---

### 📤 Sample Output

```bash
Head Brain Size Predictor using Linear Regression
Size of Dataset: (237, 4)
R² Score: 0.639311719957
```

A plot window will appear showing:
- 📉 Orange dots – Actual data points
- 📈 Green line – Linear regression prediction line

---

### 📌 Notes
The R² score indicates how well the model fits the data (closer to 1 is better).

You can experiment with new data by adding values to HeadBrain.csv.

---

### 🙌 Acknowledgements
Dataset adapted from publicly available statistical resources.

Built using:  
- scikit-learn  
- pandas  
- matplotlib
