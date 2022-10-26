from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    iris = load_iris()
    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree.fit(iris.data, iris.target)
    tree.predict(iris.data[47:53])