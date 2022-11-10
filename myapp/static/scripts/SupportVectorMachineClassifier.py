import pandas as pd
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SupportVectorMachineClassifier:
    # porciento de casos para test
    test_size = 20
    # total de datos a evaluar
    num_of_values = 5005

    def process(self, url):
        data = pd.read_csv(url)
        data.rename(columns={'CLASS_LABEL': 'phishing'}, inplace=True)

        while len(data) > self.num_of_values:
            data.drop(len(data) - 1, axis=0, inplace=True)

        float_cols = data.select_dtypes('float64').columns
        for c in float_cols:
            data[c] = data[c].astype('float32')

        int_cols = data.select_dtypes('int64').columns
        for c in int_cols:
            data[c] = data[c].astype('int32')

        sc = StandardScaler()

        x = sc.fit_transform(data.drop(['id', 'phishing'], axis=1))
        y = data['phishing']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(self.test_size / 100), shuffle=True)

        clf = svm.LinearSVC(max_iter=20000)
        clf.fit(x_train, y_train)

        y_prediction = clf.predict(x_test)

        precision = precision_score(y_test, y_prediction)
        recall = recall_score(y_test, y_prediction)
        f1 = f1_score(y_test, y_prediction)
        accuracy = accuracy_score(y_test, y_prediction)

        return [precision, recall, f1, accuracy]
