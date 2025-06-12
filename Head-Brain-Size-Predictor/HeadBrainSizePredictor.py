

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def HeadBrain():
    # Load the dataset
    dataset = pd.read_csv("HeadBrain.csv")
    print("Size of Dataset:", dataset.shape)

    # Extract features and labels
    X = dataset["Head Size(cm^3)"].values.reshape(-1, 1)
    Y = dataset["Brain Weight(grams)"].values

    # Train the Linear Regression model
    reg = LinearRegression()
    reg.fit(X, Y)
    y_pred = reg.predict(X)

    # Display R^2 score
    r2 = reg.score(X, Y)
    print("R2 Score:", r2)

    # Plotting
    plt.scatter(X, Y, color="orange", label="Actual Data")
    plt.plot(X, y_pred, color="green", label="Regression Line")
    plt.xlabel("Head Size (cm³)")
    plt.ylabel("Brain Weight (grams)")
    plt.title("Head Size vs Brain Weight")
    plt.legend()
    plt.show()

def main():
    print("Head Brain Size Predictor using Linear Regression")
    HeadBrain()
if __name__ == "__main__":
    main()
