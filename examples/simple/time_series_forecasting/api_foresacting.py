import os

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from examples.simple.time_series_forecasting.ts_pipelines import *
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from fedot.utilities.profiler.memory_profiler import MemoryProfiler
from fedot.utilities.profiler.time_profiler import TimeProfiler

datasets = {
    'australia': f'{fedot_project_root()}/examples/data/ts/australia.csv',
    'beer': f'{fedot_project_root()}/examples/data/ts/beer.csv',
    'salaries': f'{fedot_project_root()}/examples/data/ts/salaries.csv',
    'stackoverflow': f'{fedot_project_root()}/examples/data/ts/stackoverflow.csv'}


def run_ts_forecasting_example(dataset='australia', horizon: int = 30, timeout: float = None):
    time_series = pd.read_csv(datasets[dataset])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    if dataset not in ['australia']:
        idx = pd.to_datetime(time_series['idx'].values)
    else:
        # non datetime indexes
        idx = time_series['idx'].values
    time_series = time_series['value'].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input)

    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=timeout,
                  preset='fast_train',
                  composer_params={'pop_size': 10},
                  verbose_level=4,
                  _show_developer_statistics=True)

    # run AutoML model design in the same way
    pipeline = model.fit(train_data)
    pipeline.show()

    # use model to obtain forecast
    forecast = model.predict(test_data)
    target = np.ravel(test_data.target)
    print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=target))

    # plot forecasting result
    model.plot_prediction()

    return forecast


if __name__ == '__main__':
    # path = os.path.join(os.path.expanduser("~"), 'memory_profiler')
    # arguments = {'dataset': 'beer',
    #              'horizon': 30,
    #              'timeout': 0.5}
    # MemoryProfiler(run_ts_forecasting_example, kwargs=arguments, path=path,
    #                roots=[run_ts_forecasting_example], max_depth=8)

    run_ts_forecasting_example(dataset='beer', horizon=30, timeout=1)
