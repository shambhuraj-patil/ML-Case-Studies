# 📦 Wholesale Customers Case Study
This project applies clustering techniques to the Wholesale Customers dataset to identify distinct customer segments for better business decision-making.

---

## 📂 Files
WholesaleCustomer.py — 💻 Contains the full code for data preprocessing, PCA, KMeans clustering, and visualization

Wholesale Customers.csv — 📊 Dataset file

---

## 📊 Dataset
The dataset contains annual spending on various product categories for different wholesale customers.

| Channel | Region | Fresh | Milk | Grocery | Frozen | Detergents\_Paper | Delicassen |
| ------- | ------ | ----- | ---- | ------- | ------ | ----------------- | ---------- |
| 2       | 3      | 12669 | 9656 | 7561    | 214    | 2674              | 1338       |
| 2       | 3      | 7057  | 9810 | 9568    | 1762   | 3293              | 1776       |
| 2       | 3      | 6353  | 8808 | 7684    | 2405   | 3516              | 7844       |
| 1       | 3      | 13265 | 1196 | 4221    | 6404   | 507               | 1788       |
| 2       | 3      | 22615 | 5410 | 7198    | 3915   | 1777              | 5185       |

---

⚙️ Requirements
Make sure you have these Python packages installed:

    - numpy 🧮

    - pandas 📝

    - scikit-learn 🤖

    - matplotlib 📈

    - seaborn 🎨

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
