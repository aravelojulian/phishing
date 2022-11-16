import pandas as pd


def load_file(url):
    # Utiliza la librería pandas para leer el archivo
    data = pd.read_csv(url)
    # El archivo original que usamos presenta la columna CLASS_LABEL, en caso de ser así lo reemplazamos por el
    # nombre phishing
    data.rename(columns={'CLASS_LABEL': 'phishing'}, inplace=True)

    # Convertimos todas las columnas con tipo de dato float64 a float32
    float_cols = data.select_dtypes('float64').columns
    for c in float_cols:
        data[c] = data[c].astype('float32')

    # Convertimos todas las columnas con tipo de dato int64 a int32
    int_cols = data.select_dtypes('int64').columns
    for c in int_cols:
        data[c] = data[c].astype('int32')

    # Retornamos los valores del dataset
    return data
