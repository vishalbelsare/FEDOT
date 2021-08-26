import os

import numpy as np
import pandas as pd
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from data.data_manager import get_image_classification_data


def test_data_subset_correct(data_setup):
    subset_size = 50
    subset = data_setup.subset(0, subset_size - 1)

    assert len(subset.idx) == subset_size
    assert len(subset.features) == subset_size
    assert len(subset.target) == subset_size


def test_data_subset_incorrect(data_setup):
    subset_size = 105
    with pytest.raises(ValueError):
        assert data_setup.subset(0, subset_size)

    with pytest.raises(ValueError):
        assert data_setup.subset(-1, subset_size)
    with pytest.raises(ValueError):
        assert data_setup.subset(-1, -1)


def test_data_from_csv():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../../data/data/simple_classification.csv'
    task = Task(TaskTypesEnum.classification)
    df = pd.read_csv(os.path.join(test_file_path, file))
    data_array = np.array(df).T
    features = data_array[1:-1].T
    target = data_array[-1]
    idx = data_array[0]
    expected_features = InputData(features=features, target=target,
                                  idx=idx,
                                  task=task,
                                  data_type=DataTypesEnum.table).features
    actual_features = InputData.from_csv(
        os.path.join(test_file_path, file)).features
    assert np.array_equal(expected_features, actual_features)


def test_with_custom_target():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../../data/data/simple_classification.csv'
    file_custom = '../../../data/data/simple_classification_with_custom_target.csv'

    file_data = InputData.from_csv(
        os.path.join(test_file_path, file))

    expected_features = file_data.features
    expected_target = file_data.target

    custom_file_data = InputData.from_csv(
        os.path.join(test_file_path, file_custom), delimiter=';')
    actual_features = custom_file_data.features
    actual_target = custom_file_data.target

    assert not np.array_equal(expected_features, actual_features)
    assert not np.array_equal(expected_target, actual_target)

    custom_file_data = InputData.from_csv(
        os.path.join(test_file_path, file_custom), delimiter=';',
        columns_to_drop=['redundant'], target_columns='custom_target')

    actual_features = custom_file_data.features
    actual_target = custom_file_data.target

    assert np.array_equal(expected_features, actual_features)
    assert np.array_equal(expected_target, actual_target)


def test_data_from_predictions(output_dataset):
    data_1 = output_dataset
    data_2 = output_dataset
    data_3 = output_dataset
    target = output_dataset.predict
    new_input_data = InputData.from_predictions(outputs=[data_1, data_2, data_3])
    assert new_input_data.features.all() == np.array(
        [data_1.predict, data_2.predict, data_3.predict]).all()


def test_string_features_from_csv():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../../data/data/classification_with_categorical.csv'
    expected_features = InputData.from_csv(os.path.join(test_file_path, file)).features

    assert expected_features.dtype == float
    assert np.isfinite(expected_features).all()


def test_data_from_image():
    _, _, dataset_to_validate = get_image_classification_data()

    assert dataset_to_validate.data_type == DataTypesEnum.image
    assert type(dataset_to_validate.features) == np.ndarray
    assert type(dataset_to_validate.target) == np.ndarray


def test_data_from_json():
    # several features
    files_path = os.path.join('test', 'data', 'multi_modal')
    path = os.path.join(str(fedot_project_root()), files_path)
    data = InputData.from_json_files(path, fields_to_use=['votes', 'year'],
                                     label='rating', task=Task(TaskTypesEnum.regression))
    assert data.features.shape[1] == 2  # check there is two features
    assert len(data.target) == data.features.shape[0] == len(data.idx)

    # single feature
    data = InputData.from_json_files(path, fields_to_use=['votes'],
                                     label='rating', task=Task(TaskTypesEnum.regression))
    assert len(data.features.shape) == 1  # check there is one feature
    assert len(data.target) == len(data.features) == len(data.idx)


def test_multi_modal_data():
    num_samples = 5
    target = np.asarray([0, 0, 1, 0, 1])
    img_data = InputData(idx=range(num_samples),
                         features=None,  # in test the real data is not passed
                         target=target,
                         data_type=DataTypesEnum.text,
                         task=Task(TaskTypesEnum.classification))
    tbl_data = InputData(idx=range(num_samples),
                         features=None,  # in test the real data is not passed
                         target=target,
                         data_type=DataTypesEnum.table,
                         task=Task(TaskTypesEnum.classification))

    multi_modal = MultiModalData({
        'data_source_img': img_data,
        'data_source_table': tbl_data,
    })

    assert multi_modal.task.task_type == TaskTypesEnum.classification
    assert len(multi_modal.idx) == 5
    assert multi_modal.num_classes == 2
    assert np.array_equal(multi_modal.target, target)


def test_target_data_from_csv_correct():
    """ Function tests two ways of processing target columns in "from_csv"
    method
    """
    test_file_path = str(os.path.dirname(__file__))
    file = '../../../data/data/multi_target_sample.csv'
    path = os.path.join(test_file_path, file)
    task = Task(TaskTypesEnum.regression)

    # Process one column
    target_column = '1_day'
    one_column_data = InputData.from_csv(path, target_columns=target_column,
                                         columns_to_drop=['date'], task=task)

    # Process multiple target columns
    target_columns = ['1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
    seven_columns_data = InputData.from_csv(path, target_columns=target_columns,
                                            columns_to_drop=['date'], task=task)

    assert one_column_data.target.shape == (499, 1)
    assert seven_columns_data.target.shape == (499, 7)
