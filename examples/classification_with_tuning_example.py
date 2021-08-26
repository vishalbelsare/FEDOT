import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from data.data_manager import get_classification_dataset

np.random.seed(2020)


def convert_to_labels(root_operation, prediction):
    if any(root_operation == acceptable_model for acceptable_model in
           ['logit', 'lda', 'qda', 'mlp', 'svc', 'xgboost', 'bernb']):
        preds = np.array(prediction.predict)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
    else:
        preds = np.array(prediction.predict, dtype=int)

    return preds


def run_classification_tuning_experiment(pipeline, tuner=None):

    samples = [50, 550, 150]
    features = [1, 5, 10]
    classes = [2, 2, 2]
    options = [{'informative': 1, 'redundant': 0,
                'repeated': 0, 'clusters_per_class': 1},
               {'informative': 2, 'redundant': 1,
                'repeated': 1, 'clusters_per_class': 1},
               {'informative': 3, 'redundant': 1,
                'repeated': 2, 'clusters_per_class': 2}]

    for samples_amount, features_amount, \
        classes_amount, features_options in zip(samples, features, classes,
                                                options):
        print('=======================================')
        print(f'\nAmount of samples {samples_amount}, '
              f'amount of features {features_amount}, '
              f'amount of clsses {classes_amount}, '
              f'additional options {features_options}')

        x_train, y_train, x_test, y_test = get_classification_dataset(features_options,
                                                                      samples_amount,
                                                                      features_amount,
                                                                      classes_amount)
        task = Task(TaskTypesEnum.classification)

        # Prepare data to train the model
        train_input = InputData(idx=np.arange(0, len(x_train)),
                                features=x_train,
                                target=y_train,
                                task=task,
                                data_type=DataTypesEnum.table)

        predict_input = InputData(idx=np.arange(0, len(x_test)),
                                  features=x_test,
                                  target=None,
                                  task=task,
                                  data_type=DataTypesEnum.table)

        # Fit it
        pipeline.fit_from_scratch(train_input)

        # Predict
        predicted_labels = pipeline.predict(predict_input)
        preds = predicted_labels.predict

        print(f"{roc_auc(y_test, preds):.4f}\n")

        if tuner is not None:
            print(f'Start tuning process ...')

            pipeline_tuner = tuner(pipeline=pipeline, task=task,
                                iterations=50)
            tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_input,
                                                 loss_function=roc_auc)

            # Predict
            predicted_values_tuned = tuned_pipeline.predict(predict_input)
            preds_tuned = predicted_values_tuned.predict

            print(f'Obtained metrics after tuning:')
            print(f"{roc_auc(y_test, preds_tuned):.4f}\n")


# Script for testing is pipeline can process different datasets for classification
if __name__ == '__main__':
    # Prepare pipeline
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('rf', nodes_from=[node_scaling])
    pipeline_for_experiment = Pipeline(node_final)

    run_classification_tuning_experiment(pipeline=pipeline_for_experiment,
                                         tuner=PipelineTuner)
