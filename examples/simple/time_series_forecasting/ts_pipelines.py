from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def ts_ets_pipeline():
    """
    Return pipeline with the following structure:
    cut -> ets -> final forecast
    Where cut - cut part of dataset and ets - exponential smoothing
    """
    pip_builder = PipelineBuilder().add_node('cut').add_node('ets',
                                                             params={'error': 'add',
                                                                     'trend': 'add',
                                                                     'seasonal': 'add',
                                                                     'damped_trend': False,
                                                                     'seasonal_periods': 20})
    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_ets_ridge_pipeline():
    """
    Return pipeline with the following structure:
       cut -  ets \
                   -> ridge -> final forecast
    lagged - ridge /
    Where cut - cut part of dataset, ets - exponential smoothing
   """
    pip_builder = PipelineBuilder()\
        .add_sequence(('cut', {'cut_part': 0.5}),
                      ('ets', {'error': 'add', 'trend': 'add', 'seasonal': 'add',
                               'damped_trend': False, 'seasonal_periods': 20}),
                      branch_idx=0)\
        .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_glm_pipeline():
    """
    Return pipeline with the following structure:
    glm -> final forecast

    Where glm - Generalized linear model
    """
    pipeline = PipelineBuilder().add_node('glm', params={'family': 'gaussian'}).to_pipeline()
    return pipeline


def ts_glm_ridge_pipeline():
    """
    Return pipeline with the following structure:
               glm \
                   -> ridge -> final forecast
    lagged - ridge /

    Where glm - Generalized linear model
    """
    pip_builder = PipelineBuilder() \
        .add_sequence('glm', branch_idx=0) \
        .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_polyfit_pipeline(degree):
    """
    Return pipeline with the following structure:
    polyfit -> final forecast

    Where polyfit - Polynomial interpolation
    """
    pipeline = PipelineBuilder().add_node('polyfit', params={'degree': degree}).to_pipeline()
    return pipeline


def ts_polyfit_ridge_pipeline(degree):
    """
    Return pipeline with the following structure:
           polyfit \
                   -> ridge -> final forecast
    lagged - ridge /

    Where polyfit - Polynomial interpolation
    """
    pip_builder = PipelineBuilder() \
        .add_sequence(('polyfit', {'degree': degree}), branch_idx=0) \
        .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_complex_ridge_pipeline():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge -> final forecast
    lagged - ridge /
    """
    pip_builder = PipelineBuilder() \
        .add_sequence('lagged', 'rigde', branch_idx=0) \
        .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_complex_ridge_smoothing_pipeline():
    """
    Pipeline looking like this
    smoothing - lagged - ridge \
                                \
                                 ridge -> final forecast
                                /
                lagged - ridge /

    Where smoothing - rolling mean
    """
    pip_builder = PipelineBuilder() \
        .add_sequence('smoothing', 'lagged', 'ridge', branch_idx=0) \
        .add_sequence('lagged', 'ridge', branch_idx=1).join_branches('ridge')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_complex_dtreg_pipeline(first_node='lagged'):
    """
        Return pipeline with the following structure:

        lagged/sparse_lagged - dtreg \
                                        rfr -> final forecast
        lagged/sparse_lagged - dtreg /

        Where dtreg = tree regressor, rfr - random forest regressor
    """
    pip_builder = PipelineBuilder() \
        .add_sequence(first_node, 'dtreg', branch_idx=0) \
        .add_sequence(first_node, 'dtreg', branch_idx=1).join_branches('rfr')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_multiple_ets_pipeline():
    """
    Return pipeline with the following structure:
      ets
         \
    ets -> lasso -> final forecast
        /
     ets

    Where ets - exponential_smoothing
    """
    pip_builder = PipelineBuilder() \
        .add_sequence('ets', branch_idx=0) \
        .add_sequence('ets', branch_idx=1) \
        .add_sequence('ets', branch_idx=2) \
        .join_branches('lasso')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_ar_pipeline():
    """
    Return pipeline with the following structure:
    ar -> final forecast

    Where ar - auto regression
    """
    pipeline = PipelineBuilder().add_node('ar').to_pipeline()
    return pipeline


def ts_arima_pipeline():
    """
    Return pipeline with the following structure:
    arima -> final forecast

    """
    pipeline = PipelineBuilder().add_node("arima").to_pipeline()
    return pipeline


def ts_stl_arima_pipeline():
    """
    Return pipeline with the following structure:
    stl_arima -> final forecast

    """
    pipeline = PipelineBuilder().add_node("stl_arima").to_pipeline()
    return pipeline


def ts_locf_ridge_pipeline():
    """
    Pipeline with naive LOCF (last observation carried forward) model
    and lagged features
     locf  \
            ridge -> final forecast
    lagged /
    """
    pip_builder = PipelineBuilder() \
        .add_sequence('locf', branch_idx=0) \
        .add_sequence('lagged', branch_idx=1) \
        .join_branches('ridge')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def ts_naive_average_ridge_pipeline():
    """
    Pipeline with simple forecasting model (the forecast is mean value for known
    part)
    ts_naive_average \
                      ridge -> final forecast
         lagged      /
    """
    pip_builder = PipelineBuilder() \
        .add_sequence('ts_naive_average', branch_idx=0) \
        .add_sequence('lagged', branch_idx=1) \
        .join_branches('ridge')

    pipeline = pip_builder.to_pipeline()
    return pipeline


def clstm_pipeline():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge -> final forecast
             clstm /

    Where clstm - convolutional long short-term memory model
    """
    clstm_params = {'window_size': 29, 'hidden_size': 50, 'learning_rate': 0.004,
                    'cnn1_kernel_size': 5, 'cnn1_output_size': 32,
                    'cnn2_kernel_size': 4, 'cnn2_output_size': 32,
                    'batch_size': 64, 'num_epochs': 3, 'teacher_forcing': 0.8,
                    'optimizer': 'adam', 'loss': 'mse'}

    pip_builder = PipelineBuilder() \
        .add_sequence('lagged', 'ridge', branch_idx=0) \
        .add_sequence(('clstm', clstm_params), branch_idx=1) \
        .join_branches('ridge')

    pipeline = pip_builder.to_pipeline()
    return pipeline
