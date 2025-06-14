# 📦 Wholesale Customers Case Study
This project applies clustering techniques to the Wholesale Customers dataset to identify distinct customer segments for better business decision-making.

---

## 📂 Files
WholesaleCustomer.py — 💻 Contains the full code for data preprocessing, PCA, KMeans clustering, and visualization

Wholesale Customers.csv — 📊 Dataset file

---

## 📊 Dataset
The dataset contains annual spending on various product categories for different wholesale customers.

Features:

Fresh

Milk

Grocery

Frozen

Detergents_Paper

Delicassen

Removed:

Channel

Region

⚙️ Requirements
Make sure you have these Python packages installed:

numpy 🧮

pandas 📝

scikit-learn 🤖

matplotlib 📈

seaborn 🎨

## 👉 Install them with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run
Run the main script:

```bash
python WholesaleCustomer.py
```

The script will:
✅ Load and preprocess the data
✅ Apply feature scaling
✅ Reduce dimensions with PCA for visualization
✅ Use the Elbow method and Silhouette Score to find optimal clusters
✅ Apply KMeans clustering
✅ Visualize clusters

---

## 📝 Results
✨ The output includes:
    - WCSS (Elbow Method) plot to suggest optimal k
    - Silhouette Score plot
    - Final cluster visualization in 2D PCA space

---

## 📌 Notes
    - The model selects the optimal k automatically based on the Silhouette Score
    - You can modify the code to set k manually if desired
