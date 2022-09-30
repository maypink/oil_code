from pathlib import Path

import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.metrics import mean_squared_error as rmse
from sklearn.model_selection import train_test_split


class FedotWrapper:
    def __init__(self, path_to_data_dir: str):
        self.path_to_data_dir = path_to_data_dir

    def run(self, fit_type: str = 'interactive', is_visualise=True):

        data_train, data_test = self._get_data()

        auto_model = Fedot(problem='regression', seed=42, timeout=3, safe_mode=False,
                           metric='rmse', preset='best_quality', n_jobs=6, with_tuning=False)

        if fit_type == 'normal':
            auto_model.fit(features=data_train.features, target=data_train.target)
        elif fit_type == 'iterative':
            pass
        metrics_comp = auto_model.get_metrics()
        pipeline = auto_model.current_pipeline
        print(f'RMSE after composing {metrics_comp["rmse"]}')
        tuner = TunerBuilder(data_train.task) \
            .with_tuner(PipelineTuner) \
            .with_metric(RegressionMetricsEnum.RMSE) \
            .with_iterations(100) \
            .build(data_train)

        tuned_pipeline = tuner.tune(pipeline)
        tuned_prediction = tuned_pipeline.predict(data_test).predict
        metrics_tuned = rmse(data_test.target, tuned_prediction, squared=False)
        print(f'RMSE after tuning {metrics_tuned}')
        if is_visualise:
            auto_model.current_pipeline.show()

        test_pred = tuned_pipeline.predict(data_test).predict
        self.save_prediction(test_pred, 'oilcode_prediction.csv')

    def _get_data(self):
        train_x_path = Path(self.path_to_data_dir, 'x_train.csv')
        train_y_path = Path(self.path_to_data_dir, 'y_train.csv')
        # for true predict
        # test_x_path = Path(self.path_to_data_dir, 'x_test.csv')

        x_train = pd.read_csv(train_x_path).drop(['id'], axis=1)
        y_train = pd.read_csv(train_y_path).drop(['id'], axis=1)

        y_train = y_train.fillna(np.mean(y_train))

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)

        data_train = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table,
                               idx=range(len(x_train)),
                               features=x_train.values,
                               target=y_train.values)

        data_test = InputData(task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table,
                              idx=range(len(x_test)),
                              features=x_test.values,
                              target=y_test.values)
        return data_train, data_test

    @staticmethod
    def save_prediction(prediction, path_to_save):
        cols = ['id',
                'Глубина  проникания иглы при 0 °С, [мм-1]',
                'Глубина  проникания иглы при 25 °С, [мм-1]',
                'Растяжимость  при температуре 0 °С, [см]',
                'Температура размягчения, [°С]', 'Эластичность при 0 °С, [%]']
        data = pd.DataFrame(np.concatenate([np.arange(0, 10).reshape(-1, 1), prediction], axis=1),
                            columns=cols)
        data['id'] = data['id'].astype(int)
        data.to_csv(path_to_save, index=False)
