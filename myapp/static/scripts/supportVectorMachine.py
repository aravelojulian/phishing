import pandas as pd
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

num_of_values = 5005
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

SC = StandardScaler()

X = SC.fit_transform(data.drop(['id', 'phishing'], axis=1))
y = data['phishing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size / 100), shuffle=True)

clf = svm.LinearSVC(max_iter=20000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

arr = [[precision, recall, f1, accuracy]]
df = pd.DataFrame(arr, columns=['precision', 'recall', 'f1_score', 'accuracy'])
print(df)
