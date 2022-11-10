import pandas as pd


class ProcessDataset:
    # total de datos a evaluar
    num_of_values = 5005

    def load_file(self, url):
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

        return data
