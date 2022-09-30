from pathlib import Path

import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.pipelines.tuning
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.metrics import mean_squared_error as rmse


class FedotWrapper:
    def __init__(self, path_to_data_dir: str):
        self.path_to_data_dir = path_to_data_dir

    def run_oil_example(self, is_visualise=True):
        problem = 'regression'
        train_path = Path(self.path_to_data_dir, 'x_train.csv')
        test_path = Path(self.path_to_data_dir, 'x_test.csv')
        target_cols = ['Глубина  проникания иглы при 0 °С, [мм-1]',
                       'Глубина  проникания иглы при 25 °С, [мм-1]',
                       'Растяжимость  при температуре 0 °С, [см]',
                       'Температура размягчения, [°С]', 'Эластичность при 0 °С, [%]']

        train_data = InputData.from_csv(train_path,
                                        target_columns=target_cols,
                                        task=Task(TaskTypesEnum.regression))
        train, test = train_test_data_setup(train_data, split_ratio=0.75)

        val_data = InputData.from_csv(test_path,
                                      target_columns=None,
                                      task=Task(TaskTypesEnum.regression))

        auto_model = Fedot(problem=problem, seed=42, timeout=3, safe_mode=False,
                           metric='rmse', preset='best_quality', n_jobs=6, with_tuning=False)

        auto_model.fit(features=train, target=train.target)
        comp_prediction = auto_model.predict(features=test)
        metrics_comp = auto_model.get_metrics()
        pipeline = auto_model.current_pipeline
        print(f'RMSE after composing {metrics_comp["rmse"]}')
        tuner = TunerBuilder(train.task) \
            .with_tuner(PipelineTuner) \
            .with_metric(RegressionMetricsEnum.RMSE) \
            .with_iterations(100) \
            .build(train)

        tuned_pipeline = tuner.tune(pipeline)
        tuned_prediction = tuned_pipeline.predict(test).predict
        metrics_tuned = rmse(test.target, tuned_prediction, squared=False)
        print(f'RMSE after tuning {metrics_tuned}')
        if is_visualise:
            auto_model.current_pipeline.show()

        val_pred = tuned_pipeline.predict(val_data).predict
        self.save_prediction(val_pred, 'oilcode_prediction.csv')
        return metrics_tuned

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
