
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
def Iris():
    iris = load_iris()

    data = iris.data
    target = iris.target

    data_train,target_train,data_test,target_test = train_test_split(data,target,test_size=0.5)

    classifier = KNeighborsClassifier()

    classifier.fit(data_train,target_train)
    prediction = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,prediction)
    return Accuracy
def main():
    print("Iris dataset using KNN algorithm")
    Accuracy = Iris()
    print("Accuracy of dataset using KNN algorithm is ",Accuracy*100,"%")
if __name__ == "__main__":
    main()
