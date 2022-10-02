import os
from pprint import pprint
from statistics import mean

import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.model_selection import train_test_split
import numpy as np
from fedot.core.data.data import InputData
from typing import Tuple

from metric_res import get_metric, _nrmse_metric, MOMENTS, WEIGHTS


def _get_pipeline() -> Pipeline:
    """ Get pipeline with poly features, scaling and ridge regression. """
    poly_node = PrimaryNode('poly_features')
    scaling_node = SecondaryNode('scaling', nodes_from=[poly_node])
    ridge_node = SecondaryNode('ridge', nodes_from=[scaling_node])
    pipeline = Pipeline(ridge_node)
    return pipeline


def _prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Reads data from csv and splits it """
    x_train_full = pd.read_csv('data/x_train.csv').drop(['id'], axis=1)
    y_train_full = pd.read_csv('data/y_train.csv').drop(['id'], axis=1)
    REAL_X_TEST = pd.read_csv('data/x_test.csv')

    y_train_full = y_train_full.fillna(np.mean(y_train_full))

    x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full)
    return x_train, x_test, y_train, y_test, REAL_X_TEST


def _save_results(res_df: pd.DataFrame):
    """ Saves result dataframe """
    res_df['id'] = REAL_X_TEST['id']
    res_df.to_csv('oil_pred.csv', index=False)


def iterative_prediction(x_train: pd.DataFrame, x_test: pd.DataFrame,
                         y_train: pd.DataFrame, y_test: pd.DataFrame, REAL_X_TEST: pd.DataFrame):
    """ Iteratively predicts targets and adds a target to features to predict the next """
    results = []

    # init result df with columns names
    res_df = pd.DataFrame(columns=['id',
                                   'Глубина  проникания иглы при 0 °С, [мм-1]',
                                   'Глубина  проникания иглы при 25 °С, [мм-1]',
                                   'Растяжимость  при температуре 0 °С, [см]',
                                   'Температура размягчения, [°С]',
                                   'Эластичность при 0 °С, [%]'])

    for i in reversed(y_test.columns):
        print(i)
        data_train = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table,
                               idx=range(len(x_train)),
                               features=x_train.values,
                               target=y_train[i].values)

        fedot = Fedot(problem='regression', timeout=1, with_tuning=False)
        model = fedot.fit(data_train, predefined_model=_get_pipeline())

        # Tune model
        tuner = TunerBuilder(data_train.task) \
            .with_tuner(PipelineTuner) \
            .with_metric(RegressionMetricsEnum.RMSE) \
            .with_iterations(50) \
            .build(data_train)
        model = tuner.tune(model)
        data_test = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table,
                              idx=range(len(x_test)),
                              features=x_test.values,
                              target=y_test[i].values)

        data_real_test = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table,
                                   idx=range(len(x_test)),
                                   features=REAL_X_TEST.drop('id', axis=1).values)

        preds_train = model.predict(data_train).predict

        # concat dataframes
        x_train.reset_index(inplace=True, drop=True)
        x_test.reset_index(inplace=True, drop=True)
        res_df.reset_index(inplace=True, drop=True)
        x_train = pd.concat(
            [x_train, pd.DataFrame(preds_train, columns=[i])], axis=1)

        preds_test = model.predict(data_test).predict
        results.append(_nrmse_metric(preds_test, data_test.target, MOMENTS[i]) * WEIGHTS[i])

        x_test = pd.concat([x_test, pd.DataFrame(preds_test, columns=[i])], axis=1)
        preds_real_test = model.predict(data_real_test).predict
        REAL_X_TEST = pd.concat([REAL_X_TEST, pd.DataFrame(preds_real_test, columns=[i])], axis=1)
        res_df[i] = np.ravel(preds_real_test)

        _save_results(res_df=res_df)

        # print result metric
        pprint(mean(results))


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, REAL_X_TEST = _prepare_data()

    iterative_prediction(x_train, x_test, y_train, y_test, REAL_X_TEST)
