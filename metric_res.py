import pandas as pd

# Таблица со средними и среднеквадратическими отклонениями по каждому признаку

# dict(y_train.drop(['id'], axis=1).describe().loc['std'])

MOMENTS = {'Глубина  проникания иглы при 0 °С, [мм-1]': 4.1762234723302925,
           'Глубина  проникания иглы при 25 °С, [мм-1]': 11.967591280243532,
           'Растяжимость  при температуре 0 °С, [см]': 11.158831288413246,
           'Температура размягчения, [°С]': 3.5485294272609,
           'Эластичность при 0 °С, [%]': 5.716327080328637}

# Таблица с весами классов

# overall_notnull_elements = y_train.drop(['id'], axis=1).notnull().sum().sum()
# weights = {}
# for col in y_train.drop(['id'], axis=1).columns:
#     weights[col] = y_train[col].notnull().sum() / overall_notnull_elements

WEIGHTS = {'Глубина  проникания иглы при 0 °С, [мм-1]': 0.21560574948665298,
           'Глубина  проникания иглы при 25 °С, [мм-1]': 0.21971252566735114,
           'Растяжимость  при температуре 0 °С, [см]': 0.21765913757700206,
           'Температура размягчения, [°С]': 0.21765913757700206,
           'Эластичность при 0 °С, [%]': 0.1293634496919918}


def _validate_dataframe(y_pred: pd.DataFrame, y_true: pd.DataFrame):
    cols = ['id', 'Глубина  проникания иглы при 0 °С, [мм-1]',
            'Глубина  проникания иглы при 25 °С, [мм-1]',
            'Растяжимость  при температуре 0 °С, [см]',
            'Температура размягчения, [°С]', 'Эластичность при 0 °С, [%]']

    if set(y_pred.columns) != set(cols):
        raise ValueError('Неверный формат таблицы')

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError('Неверное количество строк в таблице')


def _nrmse_metric(y_pred: pd.Series, y_true: pd.Series, std: float):
    # Отбор непропущенных индексов
    rmse = ((((y_pred - y_true) / std) ** 2).sum() / y_true.shape[0]) ** (1 / 2)
    return rmse


def get_metric(pred: pd.DataFrame, true: pd.DataFrame) -> float:
    """Нормализует данные и получает взвешанную RMSE метрику
    Params
    --------
    pred : pd.DataFrame
    Таблица с предсказанными данными
    Формат таблицы совпадает с submission_example.csv
    --------
    true : pd.DataFrame
    Таблица с реальными значениями признаков
    Формат таблицы совпадает с submission_example.csv
    --------

    Returns
    --------
    metric : float
    """

    y_true = true.copy()
    y_pred = pred.copy()
    # Валидация данных
    _validate_dataframe(y_pred, y_true)

    y_true.sort_values(by='id', inplace=True)
    y_pred.sort_values(by='id', inplace=True)

    y_true.drop(['id'], axis=1, inplace=True)
    y_pred.drop(['id'], axis=1, inplace=True)

    # Вычисление weighted RMSE
    metric = 0
    for col in y_true.columns:
        metric += _nrmse_metric(y_pred[col], y_true[col], MOMENTS[col]) * WEIGHTS[col]
    return metric
