import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing


def learning(train):
    array = train.values
    x = array[1:, 0:11]  # fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides', 'free-sulphur-dioxide', 'total-sulphur-dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    y = array[1:, 11]  # quality column
    x = preprocessing.scale(x)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, test_size=0.1, random_state=1)

    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    prediction = model.predict(x_validation)
    acc = accuracy_score(y_validation, prediction)
    print(' Accuracy: {0:0.1%}'.format(acc))


if __name__ == "__main__":
    filename = ""
    names = ['fixed-acidity', 'volatile-acidity',
             'citric-acid', 'residual-sugar', 'chlorides', 'free-sulphur-dioxide', 'total-sulphur-dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    train = pd.read_csv("winequality-red.csv", names=names)
    learning(train)
