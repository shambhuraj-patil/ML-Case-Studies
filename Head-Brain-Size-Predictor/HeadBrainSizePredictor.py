
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def HeadBrain():
    dataset = pd.read_csv("HeadBrain .csv")
    print("Size of Dataset",dataset.shape)

    X = dataset["Head Size(cm^3)"].values
    Y = dataset["Brain Weight(grams)"].values

    X = X.reshape(-1,1)

    reg = LinearRegression()
    reg = reg.fit(X,Y)
    y_pred = reg.predict(X)

    r2 = reg.score(X,Y)
    print("R2 score :",r2)

    plt.scatter(X, Y, color="Orange")
    plt.plot(X, y_pred, color="Green")
    plt.xlabel("Head Size (cm^3)")
    plt.ylabel("Brain Weight (grams)")
    plt.legend(["original data","Predicted line"])
    plt.show()

def main():
    print("Head Brain Size Predictor using Linear regression")
    HeadBrain()
if __name__ == "__main__":
    main()
