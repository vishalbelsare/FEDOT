from fedot.api.main import Fedot
from test.unit.api.test_main_api import get_dataset


def test_correct_api_preprocessing():
    """ Check if dataset preprocessing was performed correctly """
    train_data, test_data, threshold = get_dataset('classification')

    fedot_model = Fedot(problem='classification', check_mode=True)
    fedot_model.fit(train_data)
