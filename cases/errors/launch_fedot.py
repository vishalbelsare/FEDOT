import timeit

import pandas as pd
import numpy as np

from fedot.api.main import Fedot


def run_classification_example(dataset_name: str, timeout: float = None):

    df = pd.read_csv(dataset_name)
    if dataset_name == 'volkert.csv':
        # Dataset volkert
        features_cols = np.array(df.columns[1:])
        features = np.array(df[features_cols])
        target = np.array(df['class'])
    elif dataset_name == 'cnae-9.csv':
        # Dataset cnae-9
        features_cols = np.array(df.columns[:-1])
        features = np.array(df[features_cols])
        target = np.array(df['Class'])
    else:
        raise ValueError()
    start_time = timeit.default_timer()

    model = Fedot(problem='classification', timeout=timeout, verbose_level=4,
                  composer_params={'pop_size': 2})
    model.fit(features=features, target=target)

    model.predict(features)
    computation_time = timeit.default_timer() - start_time
    print(f'Computation time: {computation_time}')


if __name__ == '__main__':
    run_classification_example(dataset_name='cnae-9.csv', timeout=1)
