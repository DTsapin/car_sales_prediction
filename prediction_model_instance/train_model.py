import joblib
import numpy as np
from catboost import CatBoostRegressor
from dask import dataframe as dd
from dask_ml.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import optuna

class ModelTrainer:
    """
    Класс для обучения моделей.

    ...

    Атрибуты
    ----------
    None

    Методы
    -------
    split_data:
        Метод для разделения данных на обучающую и тестовую выборки.
    preprocess_partition:
        Метод для предобработки каждого чанка (партиции в терминологии Dask) датафрейма.
    training_pipe:
        Метод пайплайна обучения модели.
    opt_models
        Метод для оптимизационного обучения модели.
    objective
        Метод внутри метода opt_models, выполняет задачу подбора гиперпараметров.
    train_on_partitions
        Метод для обучения модели по чанкам (по партициям).
    save_model:
        Метод для сохранения модели.
    predict_on_test_partitions:
        Метод для прогноза на тестовых чанках.
    compute_model_metrics:
        Метод для расчета метрик модели (заглушка для последующей интеграции с ClearML).
    """

    def split_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Метод для разделения данных на обучающую и тестовую выборки.

        Args
        ----------
        df: dd.DataFrame, Dask-датафрейм

        Returns
        -------
        train_df, test_df: tuple[dd.Dataframe], разделенный на обучение/тест датафрейм.
        """
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
        return train_df, test_df

    def preprocess_partition(self, batch: dd.DataFrame) -> dd.DataFrame:
        """
        Метод для предобработки каждого чанка (партиции) Dask-датафрейма.

        Args
        ----------
        batch: dd.Dataframe являющийся чанком Dask-датафрейма (ленивый датафрейм)

        Returns
        -------
        batch: dd.DataFrame, предобработанный чанк.
        """
        batch = batch.dropna().drop_duplicates()
        return batch
    
    def training_pipe(self, model: CatBoostRegressor, partition: dd.DataFrame, 
                      iteration: int, target: str) -> CatBoostRegressor:
        """
        Метод для формирования пайплайна обучения модели по чанкам.

        Args
        ----------
        model: CatBoostRegressor, объект модели
        partition: dd.DataFrame, чанк датафрейма
        iteration: int, порядковый номер итерации (номер чанка)
        target: целевая колонка датафрейма

        Returns
        -------
        model: CatBoostRegressor: обученная модель
        """
         # Проходим по чанкам и обучаем модель по частям
        partition = partition.compute()
        partition = self.preprocess_partition(partition)
        cat_features = [col for col in partition.columns if partition[col].dtype == "string[pyarrow]"]
        partition[cat_features] = partition[cat_features].astype("category")

        X_train = partition.drop(columns=[target])
        y_train = np.log(partition[target] + 1)

        if iteration == 0:
            model.fit(X_train, y_train, cat_features=cat_features)
        else:
            model.fit(X_train, y_train, cat_features=cat_features, init_model=model)

        return model
    
    def opt_models(
        self, train_df: dd.DataFrame, test_df: dd.DataFrame, target: str, task: str
    ) -> optuna.study.Study:
        """
        Метод для подбора гиперпараметров модели.

        Args
        ----------
        train_df: dd.DataFrame, обучающая выборка
        test_df: dd.DataFrame, тестовая выборка
        target: str, целевая колонка датафрейма
        task: str: название таска для обучения модели

        Returns
        -------
        study: optuna.study.Study, объект Optuna c best-values-ами гиперпараметров
        """

        def objective(trial: optuna.trial.Trial, train_df=train_df, test_df=test_df, target=target) -> mean_squared_error:
            """
        Метод, непосредственно выполняющий подбор гиперпараметров.

        Args
        ----------
        trial: optuna.trial.Trial, специальный optuna-объект, итерация подбора гиперпараметров
        train_df: dd.DataFrame, обучающая выборка
        test_df: dd.DataFrame, тестовая выборка
        target: str, целевая колонка датафрейма
        task: str: название таска для обучения модели

        Returns
        -------
        mean_squared_error: значение метрики MSE
            """
            params = {
            "max_depth": trial.suggest_int("max_depth", 3, 4),
            "iterations": trial.suggest_int("iterations", 300, 500, step = 50),
            "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 0.001, 1, log=True
            ),
            "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, step=0.05
                        ),
            "verbose": 0
                        }

            # Инициализируем модель с параметрами
            model = CatBoostRegressor(**params)

            # Проходим по чанкам и оптимизируем модель по частям
            partitions = train_df.to_delayed()
            for iteration, partition in enumerate(partitions):
                model = self.training_pipe(model, partition, iteration=iteration, target=target)

            # Оцениваем модель
            valid_pred, valid_actuals = self.predict_on_test_partitions(test_df, model, target)
            return mean_squared_error(valid_actuals, valid_pred) ** 0.5  # RMSE

        # Запускаем Optuna
        study = optuna.create_study(direction="minimize", study_name=task)
        study.optimize(objective, n_trials=10, n_jobs=-1)
        return study


    def train_on_partitions(self, df: dd.DataFrame, study_params: dict[str, int | float], 
                            target: str) -> CatBoostRegressor:
        """
        Метод для инициализации обучения модели по чанкам.

        Args
        ----------
        df: dd.DataFrame, Dask-датафрейм
        study_params: dict[str, int | float], словарь гиперпараметров
        target: str, целевая колонка датафрейма

        Returns
        -------
        model: CatBoostRegressor, инстанс обученной модели
        """
        params = {
                "learning_rate": study_params.best_params.get("learning_rate"),
                "l2_leaf_reg": study_params.best_params.get("l2_leaf_reg"),
                "max_depth": study_params.best_params.get("max_depth"),
                "iterations": study_params.best_params.get("iterations"),
        }
        model = CatBoostRegressor(**params, verbose=50)

        # Проходим по чанкам и обучаем модель по частям
        partitions = df.to_delayed()
        for iteration, partition in enumerate(partitions):
            model = self.training_pipe(model, partition, iteration=iteration, target=target)

        return model

    def save_model(self, model: str, model_path: str) -> None:
        """
        Метод для сохранения модели в pkl-файл.

        Args
        ----------
        model: str, название модели
        model_path: str, полный путь к файлу модели

        Returns
        -------
        None
        """
        joblib.dump(model, model_path)

    def predict_on_test_partitions(self, df: dd.DataFrame, model: str, target: str) -> tuple[np.array]:
        """
        Метод для осуществления прогнозов.

        Args
        ----------
        df: dd.DataFrame, Dask-датафрейм с данными
        model: str, название модели
        target: str, название прогнозируемой величины в датафрейме

        Returns
        -------
        np.concatenate(predictions), np.concatenate(actuals): tuple[np.array], кортеж из прогнозных и истинных данных 
        """
        
        predictions = []
        actuals = []
        for partition in df.to_delayed():
            batch = partition.compute()
            batch = self.preprocess_partition(batch)
            X_batch = batch.drop(columns=[target])
            y_batch = batch[target]
            # Обратное логарифмирование
            y_pred = np.exp(model.predict(X_batch)) - 1
            predictions.append(y_pred)
            actuals.append(y_batch)

        return np.concatenate(predictions), np.concatenate(actuals)
    
    def compute_model_metrics(self, val_true: np.array, val_preds: np.array) -> None:
        """
        Метод для расчета метрик модели (заглушка для последующей интеграции с ClearML).

        Args
        ----------
        val_true: np.array, массив истинных значений
        val_preds: np.array, массив прогнозных значений

        Returns
        -------
        None
        """
        mse = mean_squared_error(val_true, val_preds)
        rmse = (np.sqrt(mean_squared_error(val_true, val_preds)))
        mae = mean_absolute_error(val_true, val_preds)
        mape = mean_absolute_percentage_error(val_true, val_preds)
        r2 = r2_score(val_true, val_preds)
        print("Оценка тестового множества:")
        print('MSE: {:.2f}'.format(mse))
        print('RMSE: {:.2f}'.format(rmse))
        print('MAE: {:.2f}'.format(mae))
        print('MAPE: {:.2f}'.format(mape))
        print('R2: {:.2f}'.format(r2))
