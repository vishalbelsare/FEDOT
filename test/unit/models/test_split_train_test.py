import random
from copy import deepcopy

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from data.data_manager import get_synthetic_input_data
from data.pipeline_manager import generate_pipeline

np.random.seed(1)
random.seed(1)
CORRECT_MODEL_AUC_THR = 0.25


def get_roc_auc_value(pipeline: Pipeline, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = pipeline.predict(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)
    roc_auc_value_test = roc_auc(y_true=test_data.target, y_score=test_pred.predict)
    roc_auc_value_train = roc_auc(y_true=train_data.target, y_score=train_pred.predict)

    return roc_auc_value_train, roc_auc_value_test


def get_random_target_data(data: InputData) -> InputData:
    data_copy = deepcopy(data)
    data_copy.target = np.array([random.choice((0, 1)) for _ in range(len(data.target))])

    return data_copy


def get_auc_threshold(roc_auc_value: float) -> float:
    return abs(roc_auc_value - 0.5)


def test_model_fit_and_predict_correctly():
    """Checks whether the model fits and predict correctly on the synthetic dataset"""
    data = get_synthetic_input_data()

    pipeline = generate_pipeline()
    train_data, test_data = train_test_data_setup(data)

    pipeline.fit(input_data=train_data)
    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(pipeline, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert train_auc_thr >= CORRECT_MODEL_AUC_THR
    assert test_auc_thr >= CORRECT_MODEL_AUC_THR


def test_model_fit_correctly_but_predict_incorrectly():
    """Check that the model can fit the train dataset but
    can't predict the test dataset. Train and test are supposed to be
    from different distributions."""
    train_data = get_synthetic_input_data(random_state=1)
    test_data = get_synthetic_input_data(random_state=2)
    test_data.features = deepcopy(train_data.features)

    pipeline = generate_pipeline()
    pipeline.fit(input_data=train_data)
    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(pipeline, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr <= CORRECT_MODEL_AUC_THR
    assert train_auc_thr >= CORRECT_MODEL_AUC_THR


def test_model_fit_correctly_but_random_predictions_on_test():
    """Checks whether the model can fit train dataset correctly, but
    the roc_auc_score on the test dataset is close to 0.5 (predictions are random).
    Test data has not relations between features and target."""
    train_data = get_synthetic_input_data()
    test_data = get_random_target_data(train_data)

    pipeline = generate_pipeline()
    pipeline.fit(input_data=train_data)
    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(pipeline, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr <= CORRECT_MODEL_AUC_THR
    assert train_auc_thr >= CORRECT_MODEL_AUC_THR


def test_model_predictions_on_train_test_random():
    """Checks that model can't predict correctly on random train and test datasets and
    the roc_auc_scores is close to 0.5.
    Both train and test data have no relations between features and target."""
    data = get_synthetic_input_data()
    data = get_random_target_data(data)

    train_data, test_data = train_test_data_setup(data)

    pipeline = generate_pipeline()
    pipeline.fit(input_data=train_data)
    roc_auc_value_train, roc_auc_value_test = get_roc_auc_value(pipeline, train_data, test_data)
    train_auc_thr = get_auc_threshold(roc_auc_value_train)
    test_auc_thr = get_auc_threshold(roc_auc_value_test)

    assert test_auc_thr <= CORRECT_MODEL_AUC_THR
    assert train_auc_thr <= CORRECT_MODEL_AUC_THR
