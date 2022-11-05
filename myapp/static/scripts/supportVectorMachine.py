import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import HistGradientBoostingClassifier


# total de datos a evaluar
num_of_values = 5001
test_size = 20

data = pd.read_csv("../phishing.csv")

data.rename(columns={'CLASS_LABEL': 'phishing'}, inplace=True)

while len(data) > num_of_values:
    data.drop(len(data) - 1, axis=0, inplace=True)

float_cols = data.select_dtypes('float64').columns
for c in float_cols:
    data[c] = data[c].astype('float32')

int_cols = data.select_dtypes('int64').columns
for c in int_cols:
    data[c] = data[c].astype('int32')

X = data.drop(['id', 'phishing'], axis=1)
y = data['phishing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size / 100), shuffle=True)

vm = HistGradientBoostingClassifier().fit(X, y)
# vm = make_pipeline(StandardScaler(), svm.LinearSVC(max_iter=10000, loss="hinge", random_state=42))
# vm.fit(X, y)

# print(vm.score(X, y))
# print(vm.decision_function(X_test)[5])
# print(vm.predict(X_test)[5])
# y_pred = vm.predict(X_test)
#
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)

# plt.figure(figsize=(10, 5))
#
decision_function = vm.decision_function(X)
support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
support_vectors = X[X.columns[0]][support_vector_indices]

plt.subplot(1, 2, 1)
plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()

DecisionBoundaryDisplay.from_estimator(
    vm,
    X,
    ax=ax,
    grid_resolution=50,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)

plt.scatter(
    support_vectors.iloc[0],
    support_vectors.iloc[1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)

plt.title('Aymee')
plt.tight_layout()
plt.show()
