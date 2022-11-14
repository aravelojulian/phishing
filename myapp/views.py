from django.shortcuts import render
from .static.scripts.DecisionTreeClassifier import DecisionTreeClassifier
from .static.scripts.SupportVectorMachineClassifier import SupportVectorMachineClassifier
from .static.scripts.ProcessDataset import load_file


def index(request):
    # Variable donde se van a almacenar los valores que obtenemos al procesar el dataset
    dataset = []
    # Variable donde se van a almacenar los valores que obtenemos al ejecutar el algoritmo de árboles de decisión
    result_tree = []
    # Variable donde se van a almacenar los valores que obtenemos al ejecutar el algoritmo de máquinas de soporte vectorial
    result_svm = []

    # Se ejecuta cuando se detecta una acción de tipo POST en un formulario
    if request.method == "POST":
        # Obtiene el archivo del dataset para procesarlo
        dataset_file = request.FILES.get('phishing_dataset', False)

        # Procesa el dataset para obtener sus valores
        dataset = load_file(dataset_file.temporary_file_path())

        # Define el número de casos a evaluar según el algoritmo árboles de decisión
        DecisionTreeClassifier.set_num_of_values(DecisionTreeClassifier, 5005)
        # Ejecuta la clasificación según el algoritmo árboles de decisión
        result_tree = DecisionTreeClassifier.process(DecisionTreeClassifier, dataset_file.temporary_file_path())


        # Define el número de casos a evaluar según el algoritmo árboles de decisión
        SupportVectorMachineClassifier.set_num_of_values(SupportVectorMachineClassifier, 5005)
        # Ejecuta la clasificación según el algoritmo árboles de decisión
        result_svm = SupportVectorMachineClassifier.process(SupportVectorMachineClassifier, dataset_file.temporary_file_path())
    else:
        return render(request, 'index.html')

    # Envía los datos a la vista para mostrarlos
    context = {'tree': result_tree, 'dataset': dataset, 'svm': result_svm}
    return render(request, 'index.html', context)
