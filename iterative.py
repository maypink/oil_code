import os
from pprint import pprint

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

for i in y_test.columns:
    print(i)
    data_train = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table, idx=range(len(x_train)),
                           features=x_train.values,
                           target=y_train[i].values)
    fedot = Fedot(problem='regression', timeout=1, with_tuning=False)
    model = fedot.fit(data_train, predefined_model=_get_pipeline())
    tuner = TunerBuilder(data_train.task) \
        .with_tuner(PipelineTuner) \
        .with_metric(RegressionMetricsEnum.RMSE) \
        .with_iterations(50) \
        .build(data_train)
    model = tuner.tune(model)
    data_test = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table, idx=range(len(x_test)),
                          features=x_test.values,
                          target=y_test[i].values)

    data_real_test = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table,
                               idx=range(len(x_test)),
                               features=REAL_X_TEST.drop('id', axis=1).values)

    preds_train = model.predict(data_train).predict
    x_train.reset_index(inplace=True, drop=True)
    x_test.reset_index(inplace=True, drop=True)
    res.reset_index(inplace=True, drop=True)
    x_train = pd.concat(
        [x_train, pd.DataFrame(preds_train, columns=[i])], axis=1)
    preds_test = model.predict(data_test).predict

    r.append(_nrmse_metric(preds_test, data_test.target, MOMENTS[i])* WEIGHTS[i])
    x_test = pd.concat([x_test, pd.DataFrame(preds_test, columns=[i])], axis=1)
    preds_real_test = model.predict(data_real_test).predict
    REAL_X_TEST = pd.concat([REAL_X_TEST, pd.DataFrame(preds_real_test, columns=[i])], axis=1)
    res[i] = np.ravel(preds_real_test)

res['id'] = REAL_X_TEST['id']
res.to_csv('oil_pred.csv', index=False)

pprint(r)

print(sum(r))



