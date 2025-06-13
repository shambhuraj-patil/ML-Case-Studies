
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def Iris():
    iris = load_iris()
    print("Feature names of iris dataset")
    print(iris.feature_names)
    print("Target names of iris dataset")
    print(iris.target_names)

    data = iris.data
    target = iris.target

    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(data_train,target_train)
    prediction = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,prediction)
    cm = confusion_matrix(target_test,prediction)
    return Accuracy,cm
    
def main():
    print("Iris dataset using Decision tree algorithm")
    Accuracy,cm = Iris()
    print("Accuracy of dataset using Decision tree algorithm is ",Accuracy*100,"%")
    print("Confusion Matrix :",cm)
if __name__ == "__main__":
    main()