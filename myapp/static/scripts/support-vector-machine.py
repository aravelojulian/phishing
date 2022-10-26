from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


class SupportVectorMachine:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    algoritmo = SVC(kernel='linear')

    algoritmo.fit(X_train, y_train)

    y_pred = algoritmo.predict(X_test)

    matriz = confusion_matrix(y_test, y_pred)
    print('Matriz:')
    print(matriz)

    precision = precision_score(y_test, y_pred)
    print('Precisión del modelo:')
    print(precision)
