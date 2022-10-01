from pathlib import Path

import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.metrics import mean_squared_error as rmse
from metric import wnrmse


class FedotWrapper:
    def __init__(self, path_to_data_dir: str):
        self.path_to_data_dir = path_to_data_dir

    def run(self, fit_type: str = 'iterative', is_visualise=True):

        data_full, data_train, data_test, data_val = self._get_data()

        auto_model = Fedot(problem='regression', seed=42, timeout=3, safe_mode=False,
                           metric='rmse', preset='best_quality', n_jobs=6, with_tuning=False,
                           use_pipelines_cache=False, use_preprocessing_cache=False)

        # if fit_type == 'normal':
        #     auto_model.fit(features=data_train.features, target=data_train.target)
        # elif fit_type == 'iterative':
        #     pass
        pipeline = self._get_pipeline()
        pipeline.fit(data_train)
        comp_prediction = pipeline.predict(data_test).predict
        # metric_comp = auto_model.get_metrics()['rmse']
        metric_comp = rmse(data_test.target, comp_prediction, squared=False)
        wnrmse_comp = wnrmse(data_test.target, comp_prediction)
        # pipeline = auto_model.current_pipeline
        print(f'RMSE after composing {metric_comp}')
        print(f'WNRMSE after composing {wnrmse_comp}')
        tuner = TunerBuilder(data_train.task) \
            .with_tuner(PipelineTuner) \
            .with_metric(RegressionMetricsEnum.RMSE) \
            .with_iterations(100) \
            .build(data_train)

        tuned_pipeline = tuner.tune(pipeline)
        tuned_prediction = tuned_pipeline.predict(data_test).predict
        metrics_tuned = rmse(data_test.target, tuned_prediction, squared=False)
        wnrmse_tuned = wnrmse(data_test.target, tuned_prediction)
        print(f'RMSE after tuning {metrics_tuned}')
        print(f'WNRMSE after tuning {wnrmse_tuned}')
        # if is_visualise:
        #     auto_model.current_pipeline.show()

        if wnrmse_comp < wnrmse_tuned:
            pipeline.fit(data_full)
            test_pred = pipeline.predict(data_val).predict
        else:
            tuned_pipeline.fit(data_full)
            test_pred = tuned_pipeline.predict(data_val).predict
        self.save_prediction(test_pred, 'oilcode_prediction.csv')

    @staticmethod
    def _get_pipeline():
        poly_node = PrimaryNode('poly_features')
        scaling_node = SecondaryNode('scaling', nodes_from=[poly_node])
        ridge_node = SecondaryNode('ridge', nodes_from=[scaling_node])
        pipeline = Pipeline(ridge_node)
        return pipeline

    def _get_data(self):
        train_x_path = Path(self.path_to_data_dir, 'x_train.csv')
        train_y_path = Path(self.path_to_data_dir, 'y_train.csv')
        # for true predict
        val_x_path = Path(self.path_to_data_dir, 'x_test.csv')

        x_train_full = pd.read_csv(train_x_path).drop(['id'], axis=1)
        y_train_full = pd.read_csv(train_y_path).drop(['id'], axis=1)
        x_val = pd.read_csv(val_x_path).drop(['id'], axis=1)

        y_train = y_train_full.fillna(np.mean(y_train_full))

        data_full = InputData(task=Task(TaskTypesEnum.regression),
                              data_type=DataTypesEnum.table,
                              idx=range(len(x_train_full)),
                              features=x_train_full.values,
                              target=y_train.values)
        data_train, data_test = train_test_data_setup(data_full, split_ratio=0.8, shuffle_flag=True)

        data_val = InputData(task=Task(TaskTypesEnum.regression),
                             data_type=DataTypesEnum.table,
                             idx=range(len(x_val)),
                             features=x_val.values,
                             target=None)

        return data_full, data_train, data_test, data_val

    @staticmethod
    def save_prediction(prediction, path_to_save):
        cols = ['id',
                'Глубина  проникания иглы при 0 °С, [мм-1]',
                'Глубина  проникания иглы при 25 °С, [мм-1]',
                'Растяжимость  при температуре 0 °С, [см]',
                'Температура размягчения, [°С]',
                'Эластичность при 0 °С, [%]']
        data = pd.DataFrame(np.concatenate([np.arange(0, 10).reshape(-1, 1), prediction], axis=1),
                            columns=cols)
        data['id'] = data['id'].astype(int)
        data.to_csv(path_to_save, index=False)
