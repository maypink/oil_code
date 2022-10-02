import os
from pprint import pprint
from typing import Optional, List, Any

import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.model_selection import train_test_split
import numpy as np
from fedot.core.data.data import InputData
from metric import get_metric, _nrmse_metric, MOMENTS, WEIGHTS

from fedot_wrapper import _get_pipeline


def _fit_per_cluster(data: InputData, models: Optional[List[Any]] = None,
                     is_init_models: bool = False):
    if models is None:
        models = []
    clusters = np.unique(data.features[:, -1])
    for m in range(len(clusters)):
        idxs = []
        for j in range(len(data.features)):
            if data.features[j][-1] == clusters[m]:
                idxs.append(j)
        feats = [data.features[i] for i in idxs]
        targs = [data.target[i] for i in idxs]
        dataset = InputData(task=Task(TaskTypesEnum.regression),
                            data_type=DataTypesEnum.table,
                            idx=list(range(len(idxs))),
                            features=np.array([feat for feat in feats]),
                            target=np.array([targ for targ in targs]))
        if is_init_models:
            models.append(_get_pipeline())
        models[m].fit(dataset)
        # models.append(models[m])
    return models


def _predict_per_cluster(models, data: InputData):
    clusters = list(set(list(sample[-1] for sample in data.features)))
    preds = []
    for i in range(len(data.features)):
        model = models[clusters.index(data.features[i][-1])]
        input_data = InputData(task=Task(TaskTypesEnum.regression),
                               data_type=DataTypesEnum.table,
                               idx=[1],
                               features=data.features[i].reshape(1, -1),
                               target=None)
        preds.append(model.predict(input_data).predict[0])
    return preds


def _get_metric(preds, target):
    columns = ['Глубина  проникания иглы при 0 °С, [мм-1]',
               'Глубина  проникания иглы при 25 °С, [мм-1]',
               'Растяжимость  при температуре 0 °С, [см]',
               'Температура размягчения, [°С]',
               'Эластичность при 0 °С, [%]']
    id_col = pd.DataFrame(np.arange(len(preds)), columns=['id'])
    preds = id_col.join(pd.DataFrame(preds, columns=columns))
    target = id_col.join(target)
    return get_metric(preds, target)


if __name__ == '__main__':
    np.random.seed(10)
    x_train_full = pd.read_csv('data/x_train.csv').drop(['id'], axis=1)
    y_train_full = pd.read_csv('data/y_train.csv').drop(['id'], axis=1)
    REAL_X_TEST = pd.read_csv('data/x_test.csv')

    y_train_full = y_train_full.fillna(np.mean(y_train_full))

    x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full)

    r = []

    res = pd.DataFrame(columns=['id',
                                'Глубина  проникания иглы при 0 °С, [мм-1]',
                                'Глубина  проникания иглы при 25 °С, [мм-1]',
                                'Растяжимость  при температуре 0 °С, [см]',
                                'Температура размягчения, [°С]',
                                'Эластичность при 0 °С, [%]'])

    res_test = []
    is_clustering = False
    columns = [3, 4, 2, 1, 0]
    for col in columns:
        i = y_test.columns[col]
        print(i)
        data_train = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table, idx=range(len(x_train)),
                               features=x_train.values,
                               target=y_train[i].values)
        fedot = Fedot(problem='regression', timeout=1, with_tuning=False, seed=10)
        if is_clustering:
            models = _fit_per_cluster(data=data_train, is_init_models=True)
        else:
            model = fedot.fit(data_train, predefined_model=_get_pipeline())

        data_test = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table, idx=range(len(x_test)),
                              features=x_test.values,
                              target=y_test[i].values)

        data_real_test = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table,
                                   idx=range(len(x_test)),
                                   features=REAL_X_TEST.drop('id', axis=1).values)

        if is_clustering:
            preds_train = _predict_per_cluster(models, data_train)
        else:
            preds_train = model.predict(data_train).predict

        x_train.reset_index(inplace=True, drop=True)
        x_test.reset_index(inplace=True, drop=True)
        res.reset_index(inplace=True, drop=True)
        x_train = pd.concat(
            [x_train, pd.DataFrame(preds_train, columns=[i])], axis=1)
        if is_clustering:
            preds_test = _predict_per_cluster(models, data_test)
        else:
            preds_test = model.predict(data_train).predict
        res_test.append(preds_test)

        r.append(_nrmse_metric(preds_test, data_test.target, MOMENTS[i])* WEIGHTS[i])
        x_test = pd.concat([x_test, pd.DataFrame(preds_test, columns=[i])], axis=1)
        if is_clustering:
            preds_real_test = _predict_per_cluster(models, data_real_test)
        else:
            preds_real_test = model.predict(data_real_test).predict

        REAL_X_TEST = pd.concat([REAL_X_TEST, pd.DataFrame(preds_real_test, columns=[i])], axis=1)
        res[i] = np.ravel(preds_real_test)
    y_pred = np.concatenate(res_test, axis=1)
    wnrmse = _get_metric(y_pred, y_test)
    print(wnrmse)
    res['id'] = REAL_X_TEST['id']
    if is_clustering:
        res.to_csv('oil_pred_iter_clust.csv', index=False)
    else:
        res.to_csv('oil_pred_iter.csv', index=False)
