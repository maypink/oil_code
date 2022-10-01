import os
from pathlib import Path
from statistics import mean

from scipy import spatial
import numpy as np
import pandas as pd
import umap
import seaborn as sns

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
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as rmse
from sklearn.preprocessing import StandardScaler

from metric import wnrmse
from metric_res import get_metric


class FedotWrapper:
    def __init__(self, path_to_data_dir: str):
        self.path_to_data_dir = path_to_data_dir
        self.x_train_full = None
        self.y_train_full = None
        self.x_val = None

    def run(self, fit_type: str = 'iterative', is_visualise=True):

        data_full, data_train, data_test, data_val = self._get_data()

        pipeline = self._get_pipeline()
        pipeline.fit(data_train)
        comp_prediction = pipeline.predict(data_test).predict
        metric_comp = rmse(data_test.target, comp_prediction)
        # wnrmse_comp = wnrmse(data_test.target, comp_prediction)
        wnrmse_comp = self._get_wnrmse(comp_prediction)

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
        # wnrmse_tuned = wnrmse(data_test.target, tuned_prediction)
        wnrmse_tuned = self._get_wnrmse(tuned_prediction)
        print(f'RMSE after tuning {metrics_tuned}')
        print(f'WNRMSE after tuning {wnrmse_tuned}')
        if is_visualise:
            pipeline.show()

        if wnrmse_comp < wnrmse_tuned:
            pipeline.fit(data_full)
            test_pred = pipeline.predict(data_val).predict
        else:
            tuned_pipeline.fit(data_full)
            test_pred = tuned_pipeline.predict(data_val).predict
        self.save_prediction(test_pred, 'filled_data.csv')

    @staticmethod
    def _get_pipeline():
        poly_node = PrimaryNode('poly_features')
        scaling_node = SecondaryNode('scaling', nodes_from=[poly_node])
        ridge_node = SecondaryNode('ridge', nodes_from=[scaling_node])
        pipeline = Pipeline(ridge_node)
        return pipeline

    def _get_data(self):
        train_x_path = Path(self.path_to_data_dir, 'x_for_fill_train.csv')
        train_y_path = Path(self.path_to_data_dir, 'y_for_fill_train.csv')
        # for true predict
        val_x_path = Path(self.path_to_data_dir, 'x_for_fill_test.csv')

        self.x_train_full = pd.read_csv(train_x_path)
        self.y_train_full = pd.read_csv(train_y_path)
        self.x_val = pd.read_csv(val_x_path)

        y_train = self.y_train_full.fillna(np.mean(self.y_train_full))

        self.x_train_full, self.x_val = self._use_clustering(self.x_train_full, self.x_val)

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

    def _use_clustering(self, data_train: pd.DataFrame, data_val: pd.DataFrame):
        embeddings = self._get_embeddings(data_train)
        data_train = self._add_cluster_column(embeddings, data_train)
        data_val = self._add_cluster_column(embeddings, data_val)
        return data_train, data_val

    def _add_cluster_column(self, embeddings: np.ndarray, df: pd.DataFrame):
        cat_cols = ['Адгезионная добавка', 'Полимер']
        x_train_umap = df.drop(columns=['Пластификатор'])
        ohe = pd.get_dummies(x_train_umap[cat_cols])
        x_train_umap = x_train_umap.drop(columns=cat_cols).join(ohe)
        scaled_sample = StandardScaler().fit_transform(x_train_umap)
        reducer = umap.UMAP()
        cur_embeddings = reducer.fit_transform(scaled_sample)
        emb_column = []
        emb_mean_0 = mean([emb[1] for emb in embeddings])
        for i in range(x_train_umap.shape[0]):
            # cosine_dists = [np.linalg.norm(cur_embeddings[i]-emb) for emb in embeddings]
            cosine_dists = [spatial.distance.cosine(cur_embeddings[i], emb) for emb in embeddings]
            emb_with_min_cos = embeddings[cosine_dists.index(min(cosine_dists))]
            cur_emb_value = 'cluster0' if emb_with_min_cos[1] < emb_mean_0 else 'cluster1'
            emb_column.append(cur_emb_value)
        df['embedding'] = np.array(emb_column)
        return df

    @staticmethod
    def _get_embeddings(data: pd.DataFrame):
        cat_cols = ['Адгезионная добавка', 'Полимер']
        x_train_umap = data.drop(columns=['Пластификатор'])
        ohe = pd.get_dummies(x_train_umap[cat_cols])
        x_train_umap = x_train_umap.drop(columns=cat_cols).join(ohe)
        scaled_data = StandardScaler().fit_transform(x_train_umap)

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(scaled_data)
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the oil dataset', fontsize=24)
        # plt.show()
        return embedding

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

    def _get_wnrmse(self, prediction: np.ndarray):
        y_cols = ['Глубина  проникания иглы при 0 °С, [мм-1]',
                  'Глубина  проникания иглы при 25 °С, [мм-1]',
                  'Растяжимость  при температуре 0 °С, [см]',
                  'Температура размягчения, [°С]', 'Эластичность при 0 °С, [%]']
        train_df = self.x_train_full.join(self.y_train_full)
        test_df = self.x_val.join(pd.DataFrame(prediction, columns=['Эластичность при 0 °С, [%]']))
        data = pd.concat([train_df, test_df]).reset_index(drop=True)
        id_col = pd.DataFrame(np.arange(len(data)), columns=['id'])
        y_train = id_col.join(data[y_cols])
        y_train['id'] = y_train['id'].astype(int)
        metric = get_metric(y_train, pd.read_csv(os.path.join(self.path_to_data_dir, "y_train.csv")))
        return metric
