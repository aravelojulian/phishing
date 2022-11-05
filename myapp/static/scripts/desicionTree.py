import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import classification_report

# número de rasgos a evaluar
num_of_features = 48
# total de datos a evaluar
num_of_values = 5005

pd.set_option('display.max_columns', None)

data = pd.read_csv("../phishing.csv")

while len(data) > num_of_values:
    data.drop(len(data) - 1, axis=0, inplace=True)

float_cols = data.select_dtypes('float64').columns
for c in float_cols:
    data[c] = data[c].astype('float32')

int_cols = data.select_dtypes('int64').columns
for c in int_cols:
    data[c] = data[c].astype('int32')

data.rename(columns={'CLASS_LABEL': 'phishing'}, inplace=True)

data['phishing'].value_counts().plot(kind='bar')

X = data.drop(['id', 'phishing'], axis=1)
y = data['phishing']

discrete_features = X.dtypes == int

my_phishing = mutual_info_classif(X, y, discrete_features=discrete_features)
my_phishing = pd.Series(my_phishing, name='My Phishing', index=X.columns)
my_phishing = my_phishing.sort_values(ascending=False)

arr = []

top_n_features = my_phishing.sort_values(ascending=False).head(num_of_features).index.tolist()
X = data[top_n_features]
y = data['phishing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

lr = tree.DecisionTreeClassifier()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
print()

tree.plot_tree(lr)
plt.show()

arr.append([num_of_features, precision, recall, f1, accuracy])

df = pd.DataFrame(arr, columns=['num_of_features', 'precision', 'recall', 'f1_score', 'accuracy'])
print(df)
