import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def wine_predictor():
    dataset = pd.read_csv("WinePredictor.csv")
    print(dataset.head())

    x = dataset.drop("Class", axis=1)
    y = dataset["Class"]
    
    # Visualization 
    print("Visualizing pairplot...")
    sns.pairplot(dataset, hue='Class', diag_kind='kde', markers=["o", "s", "D"])
    plt.title("Pairplot of Features")
    plt.savefig(r"C:\Users\shamb\Desktop\self\wine_dataset.png")
    plt.show()
    
    # Split the data into training and testing
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

    # Feature scaling for scaling the training features
    sc = StandardScaler()
    scaled_x_train = sc.fit_transform(xtrain)
    scaled_x_test = sc.transform(xtest)
    
    # Initialize Logistic Regression model
    reg = LogisticRegression(max_iter=10000)
    
    # Train the model with scaled training data
    reg.fit(scaled_x_train, ytrain)
    
    # Test the model using scaled test data
    prediction = reg.predict(scaled_x_test)  # Use numpy array directly for prediction
    
    # Calculate accuracy
    accuracy = accuracy_score(ytest, prediction)
    print("Accuracy using Logistic Regression:", accuracy * 100, "%")

    # Custom input
    custom_input = [[13.24, 2.59, 2.87, 21.0, 118, 2.80, 2.69, 0.39, 1.82, 4.32, 1.04, 2.93, 735]]
    
    # Scale custom input for prediction
    custom_input_scaled = sc.transform(custom_input)
    custom_prediction = reg.predict(custom_input_scaled)
    print("Prediction for the provided custom input:", custom_prediction[0])

def main():
    print("Wine predictor")
    wine_predictor()

if __name__ == "__main__":
    main()
