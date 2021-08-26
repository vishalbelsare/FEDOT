import numpy as np

from fedot.core.data.data_split import train_test_data_setup
from test.unit.models.test_split_train_test import get_roc_auc_value
from data.data_manager import get_synthetic_input_data
from data.pipeline_manager import generate_pipeline


def test_pipeline_with_clusters_fit_correct():
    mean_roc_on_test = 0

    # mean ROC AUC is analysed because of stochastic clustering
    for _ in range(5):
        data = get_synthetic_input_data(n_samples=10000)

        pipeline = generate_pipeline()
        train_data, test_data = train_test_data_setup(data)

        pipeline.fit(input_data=train_data)
        _, roc_on_test = get_roc_auc_value(pipeline, train_data, test_data)
        mean_roc_on_test = np.mean([mean_roc_on_test, roc_on_test])

    roc_threshold = 0.5
    assert mean_roc_on_test > roc_threshold
