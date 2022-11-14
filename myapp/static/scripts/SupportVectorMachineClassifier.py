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

    # Cambia la cantidad de datos a evaluar
    def set_num_of_values(self, num):
        self.num_of_values = num

    # Cambia el tamaño de la prueba
    def set_test_size(self, num):
        self.test_size = num

    # Comienza el proceso de clasificación con el Algoritmo Máquinas de soporte vectorial
    def process(self, url):
        # Utiliza la librería pandas para leer el archivo
        data = pd.read_csv(url)
        # El archivo original que usamos presenta la columna CLASS_LABEL, en caso de ser así lo reemplazamos por el
        # nombre phishing
        data.rename(columns={'CLASS_LABEL': 'phishing'}, inplace=True)

        # Eliminamos las filas sobrantes con respecto al número de casos a evaluar
        while len(data) > self.num_of_values:
            data.drop(len(data) - 1, axis=0, inplace=True)

        # Convertimos todas las columnas con tipo de dato float64 a float32
        float_cols = data.select_dtypes('float64').columns
        for c in float_cols:
            data[c] = data[c].astype('float32')

        # Convertimos todas las columnas con tipo de dato int64 a int32
        int_cols = data.select_dtypes('int64').columns
        for c in int_cols:
            data[c] = data[c].astype('int32')

        # Limpiamos el dataset para evitar columnas con valores NAN
        sc = StandardScaler()

        # Eliminamos las columnas id y phishing para utilizar el resto del rango como datos para el algoritmo
        x = sc.fit_transform(data.drop(['id', 'phishing'], axis=1))

        # Nos quedamos solo con la columna phishing
        y = data['phishing']

        # Dividimos nuestra muestra para entrenamiento y prueba, utilizamos un tamaño de prueba definido y lo hacemos
        # ramdom
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(self.test_size / 100), shuffle=True)

        # Instanciamos el algoritmo de clasificación utilizando la función Linear de las máquinas de soporte vectorial
        clf = svm.LinearSVC(max_iter=20000)
        # Ejecutamos el algoritmo con las muestras de entrenamiento
        clf.fit(x_train, y_train)

        # Ejecutamos las predicciones sobre la muestra de prueba
        y_prediction = clf.predict(x_test)

        # Hallamos la precisión del algoritmo para esta muestra de entrenamiento y prueba (Recordar que para el mismo
        # juego de datos podría dar un poco diferente, ya que escogemos la muestra de prueba ramdom)
        precision = precision_score(y_test, y_prediction)
        # Hallamos la exhaustividad del algoritmo
        recall = recall_score(y_test, y_prediction)
        # Hallamos el valor F del algoritmo
        f1 = f1_score(y_test, y_prediction)
        # Hallamos la efectividad del algoritmo
        accuracy = accuracy_score(y_test, y_prediction)

        # Retornamos una matriz con los resultados obtenidos
        return [precision, recall, f1, accuracy]
