
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def customer():
    file_path = r"C:\Users\shamb\Downloads\Wholesale Customers.csv"

    if not os.path.exists(file_path):
        print("Error: File not found. Please check the file path.")
        return

    dataset = pd.read_csv(file_path)

    print("First 5 rows from loaded dataset :")
    print(dataset.head())

    print("Check for missing values :")
    print(dataset.isnull().sum())

    # Drop unnecessary columns
    dataset.drop(columns=["Channel", "Region"], inplace=True)
    print("Dataset after dropping unnecessary columns :")
    print(dataset.head())

    # Scale features
    sc = StandardScaler()
    scaled_dataset = sc.fit_transform(dataset)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_dataset)
    X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

    # Plot before clustering
    sns.scatterplot(data=X_pca_df, x='PC1', y='PC2')
    plt.title('PCA of Data (Before Clustering)')
    plt.show()

    # Elbow Method
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, n_init=10, random_state=42)
        km.fit(scaled_dataset)
        wcss.append(km.inertia_)

    plt.plot(range(1, 11), wcss, marker="o")
    plt.xlabel("No of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(range(1, 11))
    plt.grid(axis="x")
    plt.show()

    # Silhouette scores for different k
    silhouette_scores = []
    cluster_range = range(2, 15)

    for k in cluster_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scaled_dataset)
        score = silhouette_score(scaled_dataset, labels)
        silhouette_scores.append(score)

    # Plot Silhouette Score vs k
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker="o", linestyle="-", label="Silhouette Score")

    # Mark all points and annotate
    for k, score in zip(cluster_range, silhouette_scores):
        plt.scatter(k, score, color='blue', s=60)
        plt.text(k, score + 0.01, f"{score:.2f}", ha='center')

    # Highlight best k
    best_k_index = np.argmax(silhouette_scores)
    best_k = cluster_range[best_k_index]
    best_score = silhouette_scores[best_k_index]
    plt.scatter(best_k, best_score, color='red', s=120, label=f"Best k = {best_k}")

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Method for Optimal k")
    plt.legend()
    plt.grid(axis="x")
    plt.xticks(cluster_range)
    plt.show()

    print("Silhouette Scores for each k:")
    for k, score in zip(cluster_range, silhouette_scores):
        print(f"k = {k}, Silhouette Score = {score:.4f}")

    print(f"\nOptimal number of clusters based on Silhouette Score: {best_k}")

    # Final KMeans with best_k
    km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    dataset["Cluster"] = km_final.fit_predict(scaled_dataset)

    print(f"\nDataset with {best_k} Cluster Assignments:")
    print(dataset.head())

    # Visualize clusters
    X_pca_df["Cluster"] = dataset["Cluster"]
    sns.scatterplot(data=X_pca_df, x="PC1", y="PC2", hue="Cluster", palette="viridis")
    plt.title("Clusters Visualized After K-Means")
    plt.show()

customer()
