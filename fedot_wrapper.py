import os
from pathlib import Path
from statistics import mean
from typing import Optional, List, Any

from scipy import spatial
import numpy as np
import pandas as pd
#import umap
import sklearn.cluster as cluster
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

    def run(self, fit_type: str = 'iterative', is_visualise: bool = False, is_clustering: bool = True):

        data_full, data_train, data_test, data_val = self._get_data()

        if is_clustering:
            models = self._fit_per_cluster(data=data_train, is_init_models=True)
            comp_prediction = self._predict_per_cluster(models, data_test)
        else:
            pipeline = self._get_pipeline()
            pipeline.fit(data_train)
            comp_prediction = pipeline.predict(data_test).predict

        metric_comp = rmse(data_test.target, comp_prediction, squared=False)
        # wnrmse_comp = wnrmse(data_test.target, comp_prediction)
        wnrmse_comp = self._get_wnrmse(data_test.target, comp_prediction)

        print(f'RMSE after composing {metric_comp}')
        print(f'WNRMSE after composing {wnrmse_comp}')

        tuner = TunerBuilder(data_train.task) \
            .with_tuner(PipelineTuner) \
            .with_metric(RegressionMetricsEnum.RMSE) \
            .with_iterations(30) \
            .build(data_train)

        if is_clustering:
            tuned_models = []
            for model in models:
                tuned_models.append(tuner.tune(model))
            # self._fit_per_cluster(models=tuned_models, data=data_test)
            tuned_prediction = self._predict_per_cluster(models=tuned_models, data=data_test)
        else:
            tuned_pipeline = tuner.tune(pipeline)
            tuned_prediction = tuned_pipeline.predict(data_test).predict

        metrics_tuned = rmse(data_test.target, tuned_prediction, squared=False)
        # wnrmse_tuned = wnrmse(data_test.target, tuned_prediction)
        wnrmse_tuned = self._get_wnrmse(data_test.target, tuned_prediction)
        print(f'RMSE after tuning {metrics_tuned}')
        print(f'WNRMSE after tuning {wnrmse_tuned}')
        # if is_visualise:
        #     pipeline.show()

        if wnrmse_comp < wnrmse_tuned:
            if is_clustering:
                models = self._fit_per_cluster(data=data_full, models=models)
                test_pred = self._predict_per_cluster(models, data_val)
            else:
                pipeline.fit(data_full)
                test_pred = pipeline.predict(data_val).predict
        else:
            if is_clustering:
                tuned_models2 = []
                for model in models:
                    tuned_models2.append(tuner.tune(model))
                tuned_models2 = self._fit_per_cluster(data=data_full, models=tuned_models2)
                test_pred = self._predict_per_cluster(tuned_models2, data_val)
            else:
                tuned_pipeline.fit(data_full)
                test_pred = tuned_pipeline.predict(data_val).predict

        self.save_prediction(test_pred, 'filled_data.csv')

    def _fit_per_cluster(self, data: InputData, models: Optional[List[Any]] = None,
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
                                features=np.array([list(feat) for feat in feats]),
                                target=np.array([list(feat) for feat in targs]))
            if is_init_models:
                models.append(self._get_pipeline())
            models[m].fit(dataset)
            # models.append(models[m])
        return models

    @staticmethod
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

    @staticmethod
    def _get_pipeline():
        poly_node = PrimaryNode('poly_features')
        scaling_node = SecondaryNode('scaling', nodes_from=[poly_node])
        ridge_node = SecondaryNode('ridge', nodes_from=[scaling_node])
        pipeline = Pipeline(ridge_node)
        return pipeline

    def _get_data(self):
        train_x_path = Path(self.path_to_data_dir, 'x_train_clustered.csv')
        train_y_path = Path(self.path_to_data_dir, 'y_train.csv')
        # for true predict
        val_x_path = Path(self.path_to_data_dir, 'x_test_clustered.csv')

        self.x_train_full = pd.read_csv(train_x_path)
        self.y_train_full = pd.read_csv(train_y_path)
        self.x_val = pd.read_csv(val_x_path)

        y_train = self.y_train_full.fillna(np.mean(self.y_train_full))

        #self.x_train_full, self.x_val = self._use_clustering(self.x_train_full, self.x_val)

        data_full = InputData(task=Task(TaskTypesEnum.regression),
                              data_type=DataTypesEnum.table,
                              idx=range(len(self.x_train_full)),
                              features=self.x_train_full.drop(columns=['id']).values,
                              target=y_train.drop(columns=['id']).values)
        data_train, data_test = train_test_data_setup(data_full, split_ratio=0.75, shuffle_flag=True)

        data_val = InputData(task=Task(TaskTypesEnum.regression),
                             data_type=DataTypesEnum.table,
                             idx=range(len(self.x_val)),
                             features=self.x_val.drop(columns=['id']).values,
                             target=None)

        return data_full, data_train, data_test, data_val

    def _use_clustering(self, data_train: pd.DataFrame, data_val: pd.DataFrame):
        embeddings, labels = self._get_embeddings(data_train)
        data_train = self._add_cluster_column(embeddings, data_train, labels)
        data_val = self._add_cluster_column(embeddings, data_val, labels)
        return data_train, data_val

    # @staticmethod
    # def _add_cluster_column(embeddings: np.ndarray, df: pd.DataFrame, labels):
    #     cat_cols = ['Адгезионная добавка', 'Полимер']
    #     x_train_umap = df.drop(columns=['Пластификатор'])
    #     ohe = pd.get_dummies(x_train_umap[cat_cols])
    #     x_train_umap = x_train_umap.drop(columns=cat_cols).join(ohe)
    #     scaled_sample = StandardScaler().fit_transform(x_train_umap)
    #     reducer = umap.UMAP()
    #     cur_embeddings = reducer.fit_transform(scaled_sample)
    #     emb_column = []
    #     for i in range(x_train_umap.shape[0]):
    #         # cosine_dists = [np.linalg.norm(cur_embeddings[i]-emb) for emb in embeddings]
    #         cosine_dists = [spatial.distance.cosine(cur_embeddings[i], emb) for emb in embeddings]
    #         label = labels[cosine_dists.index(min(cosine_dists))]
    #         cur_emb_value = 'cluster0' if label == 0 else 'cluster1'
    #         emb_column.append(cur_emb_value)
    #     df['cluster'] = np.array(emb_column)
    #     return df

    @staticmethod
    def _get_embeddings(data: pd.DataFrame):
        cat_cols = ['Адгезионная добавка', 'Полимер']
        x_train_umap = data.drop(columns=['Пластификатор'])
        ohe = pd.get_dummies(x_train_umap[cat_cols])
        x_train_umap = x_train_umap.drop(columns=cat_cols).join(ohe)
        scaled_data = StandardScaler().fit_transform(x_train_umap)

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(scaled_data)
        kmeans_labels = cluster.KMeans(n_clusters=2).fit_predict(scaled_data)
        emb_mean = mean([emb[0] for emb in embedding])
        # colors = ['red' if emb[0] > emb_mean else 'green' for emb in embedding]

        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=kmeans_labels)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('Projection of the oil dataset', fontsize=24)
        plt.grid()
        plt.show()
        return embedding, kmeans_labels

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
        # train_df = self.x_train_full.join(self.y_train_full)
        # test_df = self.x_val.join(pd.DataFrame(prediction, columns=['Эластичность при 0 °С, [%]']))
        # data = pd.concat([train_df, test_df]).reset_index(drop=True)
        id_col = pd.DataFrame(np.arange(len(prediction)), columns=['id'])
        pred_df = id_col.join(pd.DataFrame(prediction, columns=y_cols))
        pred_df.to_csv('oilcode_prediction.csv', index=False)
        # x_train = id_col.join(data[x_cols])
        # y_train = id_col.join(data[y_cols])
        # x_train['id'] = x_train['id'].astype(int)
        # y_train['id'] = y_train['id'].astype(int)
        # x_train.to_csv('x_train_filled.csv', index=False)
        # y_train.to_csv('y_train_filled.csv', index=False)

    def _get_wnrmse(self, target: np.ndarray, prediction: np.ndarray):
        y_cols = ['Глубина  проникания иглы при 0 °С, [мм-1]',
                  'Глубина  проникания иглы при 25 °С, [мм-1]',
                  'Растяжимость  при температуре 0 °С, [см]',
                  'Температура размягчения, [°С]', 'Эластичность при 0 °С, [%]']
        id_col = pd.DataFrame(np.arange(len(target)), columns=['id'])
        id_col['id'] = id_col['id'].astype(int)
        y_true = id_col.join(pd.DataFrame(target, columns=y_cols))
        y_pred = id_col.join(pd.DataFrame(prediction, columns=y_cols))
        # train_df = self.x_train_full.join(self.y_train_full.drop(columns=['id']))
        # test_df = self.x_val.join(pd.DataFrame(prediction, columns=y_cols))
        # data = pd.concat([train_df, test_df]).reset_index(drop=True)
        # id_col = pd.DataFrame(np.arange(len(data)), columns=['id'])
        # y_train = id_col.join(data[y_cols])
        # y_train['id'] = y_train['id'].astype(int)
        metric = get_metric(y_pred, y_true)
        return metric
