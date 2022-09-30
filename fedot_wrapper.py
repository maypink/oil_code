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
from metric import wnrmse


class FedotWrapper:
    def __init__(self, path_to_data_dir: str):
        self.path_to_data_dir = path_to_data_dir
        self.x_train_full = None
        self.y_train_full = None
        self.x_val = None

    def run(self, fit_type: str = 'iterative', is_visualise=True):

        data_full, data_train, data_test, data_val = self._get_data()

        auto_model = Fedot(problem='regression', seed=42, timeout=3, safe_mode=False,
                           metric='rmse', preset='best_quality', n_jobs=6, with_tuning=False,
                           use_pipelines_cache=False, use_preprocessing_cache=False)

        if fit_type == 'normal':
            auto_model.fit(features=data_train.features, target=data_train.target,
                           predefined_model='auto')
        elif fit_type == 'iterative':
            pass
        comp_prediction = auto_model.predict(features=data_test)
        metric_comp = auto_model.get_metrics()['rmse']
        wnrmse_comp = wnrmse(data_test.target, comp_prediction)
        pipeline = auto_model.current_pipeline
        print(f'RMSE after composing {metric_comp}')
        print(f'WNRMSE after composing {wnrmse_comp}')
        tuner = TunerBuilder(data_train.task) \
            .with_tuner(PipelineTuner) \
            .with_metric(RegressionMetricsEnum.RMSE) \
            .with_iterations(30) \
            .build(data_train)

        tuned_pipeline = tuner.tune(pipeline)
        tuned_prediction = tuned_pipeline.predict(data_test).predict
        metrics_tuned = rmse(data_test.target, tuned_prediction, squared=False)
        wnrmse_tuned = wnrmse(data_test.target, tuned_prediction)
        print(f'RMSE after tuning {metrics_tuned}')
        print(f'WNRMSE after tuning {wnrmse_tuned}')
        if is_visualise:
            auto_model.current_pipeline.show()

        if wnrmse_comp < wnrmse_tuned:
            auto_model.current_pipeline.fit(data_full)
            test_pred = auto_model.current_pipeline.predict(data_val).predict
        else:
            tuned_pipeline.fit(data_full)
            test_pred = tuned_pipeline.predict(data_val).predict
        self.save_prediction(test_pred, 'filled_data.csv')

    def _get_data(self):
        train_x_path = Path(self.path_to_data_dir, 'x_for_fill_train.csv')
        train_y_path = Path(self.path_to_data_dir, 'y_for_fill_train.csv')
        # for true predict
        val_x_path = Path(self.path_to_data_dir, 'x_for_fill_test.csv')

        self.x_train_full = pd.read_csv(train_x_path)
        self.y_train_full = pd.read_csv(train_y_path)
        self.x_val = pd.read_csv(val_x_path)

        y_train = self.y_train_full.fillna(np.mean(self.y_train_full))

        data_full = InputData(task=Task(TaskTypesEnum.regression),
                              data_type=DataTypesEnum.table,
                              idx=range(len(self.x_train_full)),
                              features=self.x_train_full.drop(columns=['Полимер']).values,
                              target=y_train.values)
        data_train, data_test = train_test_data_setup(data_full, split_ratio=0.75, shuffle_flag=True)

        data_val = InputData(task=Task(TaskTypesEnum.regression),
                             data_type=DataTypesEnum.table,
                             idx=range(len(self.x_val)),
                             features=self.x_val.drop(columns=['Полимер']).values,
                             target=None)

        return data_full, data_train, data_test, data_val

    def save_prediction(self, prediction, path_to_save):
        x_cols = ['% массы <Адгезионная добавка>', '% массы <Базовый битум>',
                '% массы <Пластификатор>', '% массы <Полимер>',
                '% массы <Сшивающая добавка>', 'Исходная игла при 25С <Базовый битум>',
                'Адгезионная добавка', 'Пластификатор', 'Полимер',
                'Базовая пенетрация для расчёта пластификатора',
                'Расчёт рецептуры на глубину проникания иглы при 25']
        y_cols = ['Глубина  проникания иглы при 0 °С, [мм-1]',
                  'Глубина  проникания иглы при 25 °С, [мм-1]',
                  'Растяжимость  при температуре 0 °С, [см]',
                  'Температура размягчения, [°С]', 'Эластичность при 0 °С, [%]']
        train_df = self.x_train_full.join(self.y_train_full)
        test_df = self.x_val.join(pd.DataFrame(prediction, columns=['Эластичность при 0 °С, [%]']))
        data = pd.concat([train_df, test_df]).reset_index(drop=True)
        id_col = pd.DataFrame(np.arange(len(data)), columns=['id'])
        x_train = id_col.join(data[x_cols])
        y_train = id_col.join(data[y_cols])
        x_train['id'] = x_train['id'].astype(int)
        y_train['id'] = y_train['id'].astype(int)
        x_train.to_csv('x_train_filled.csv', index=False)
        y_train.to_csv('y_train_filled.csv', index=False)