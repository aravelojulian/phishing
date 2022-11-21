import pandas as pd
from django.shortcuts import render

from .static.scripts.DecisionTreeClassifier import DecisionTreeClassifier
from .static.scripts.ProcessDataset import load_file
from .static.scripts.SupportVectorMachineClassifier import SupportVectorMachineClassifier


class Dataset:
    url = ''
    dataset = pd.DataFrame()

    def set_url(self, new_url):
        self.url = new_url

    def load_data(self):
        self.dataset = load_file(self.url)

    def assign_column(self, column_name, column_data):
        self.dataset.insert(len(self.dataset), column_name, column_data, True)

    def get_values(self):
        return self.dataset.to_records()


url = 'C:/Users/Ale/Documents/Projects/phish/myapp/static/phishing.csv'
dataset_class = Dataset()


def generate_report(request):
    dataset_class.dataset.to_excel(r'Reporte.xlsx', index=False)

    return render(request, "report.html")


def details(request, row):
    data = dataset_class.dataset.loc[dataset_class.dataset['id'] == int(float(row))]
    data = data.to_records()[0]

    return render(request, "details.html", {'row': data})


def index(request):
    # Creamos una instancia de la clase DecisionTreeClassifier
    tree = DecisionTreeClassifier()

    # Creamos una instancia de la clase SupportVectorMachineClassifier
    svm = SupportVectorMachineClassifier()

    # Define el número de casos a evaluar según el algoritmo árboles de decisión
    tree.set_num_of_values(5005)
    # Ejecuta la clasificación según el algoritmo árboles de decisión y guarda el resultado
    result_tree = tree.process(url)

    # Define el número de casos a evaluar según el algoritmo árboles de decisión
    svm.set_num_of_values(5005)
    # Ejecuta la clasificación según el algoritmo árboles de decisión y guarda el resultado
    result_svm = svm.process(url)

    # Se ejecuta cuando se detecta una acción de tipo POST en un formulario
    if request.method == "POST":
        # Obtiene el archivo del dataset para procesarlo
        dataset_file = request.FILES.get('phishing_dataset', False)

        # Guarda la url del dataset
        dataset_class.set_url(dataset_file.temporary_file_path())

        # Procesa el dataset para obtener sus valores
        dataset_class.load_data()

        # Ejecutamos la predicción del dataset con el Algoritmo Árboles de Decisión
        predict_tree = tree.predict(dataset_class.dataset)

        # Ejecutamos la predicción del dataset con el Algoritmo Máquinas de Soporte Vectorial
        predict_svm = svm.predict(dataset_class.dataset)

        # Agregamos una nueva columna con los valores de la predicción del algoritmo Algoritmo Árboles de Decisión
        dataset_class.assign_column('TreeClassification', predict_tree)

        # Agregamos una nueva columna con los valores de la predicción del algoritmo Máquinas de Soporte Vectorial
        dataset_class.assign_column('SVMClassification', predict_svm)

        # Envía los datos a la vista para mostrarlos
        context = {'tree': result_tree, 'svm': result_svm, 'dataset': dataset_class.get_values()}
        return render(request, 'index.html', context)
    else:
        # Envía los datos a la vista para mostrarlos
        context = {'tree': result_tree, 'svm': result_svm}
        return render(request, 'index.html', context)
