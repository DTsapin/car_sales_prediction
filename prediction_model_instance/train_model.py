import joblib
import numpy as np
from catboost import CatBoostRegressor
from dask import dataframe as dd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import optuna

class ModelTrainer:
    # TODO названия модели - в таск, аналогично привязка к таргетам

    def split_data(self, df: dd.DataFrame):
        """Разделяем данные на обучающую и тестовую выборки"""
        train_df = df.partitions[:-1]
        test_df = df.partitions[-1]
        return train_df, test_df

    def preprocess_partition(self, batch):
        """Обрабатываем каждую партицию: удаляем пропуски, дубликаты"""
        batch = batch.dropna().drop_duplicates()
        return batch
    
    def training_pipe(self, model, partition, i, target):
         # Проходим по чанкам и обучаем модель по частям
        partition = partition.compute()  # Загружаем партицию в RAM
        partition = self.preprocess_partition(partition)
        cat_features = [col for col in partition.columns if partition[col].dtype == "string[pyarrow]"]
        partition[cat_features] = partition[cat_features].astype("category")

        X_train = partition.drop(columns=[target])
        y_train = np.log(partition[target] + 1)

        if i == 0:
            model.fit(X_train, y_train, cat_features=cat_features)
        else:
            model.fit(X_train, y_train, cat_features=cat_features, init_model=model)

        return model
    
    def opt_models(
        self, train_df: dd.DataFrame, test_df: dd.DataFrame, target, task
    ) -> optuna.study.Study:
        """Блок подбора гиперпараметров"""

        def objective(trial, train_df=train_df, target=target):
        # Формируем словарь гиперпараметров TODO разобраться с дублирующимся кодом
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
            for i, partition in enumerate(partitions):
                model = self.training_pipe(model, partition, i=i, target=target)

            # Оцениваем модель
            valid_pred, valid_actuals = self.predict_on_test_partitions(test_df, model, target)
            return mean_squared_error(valid_actuals, valid_pred) ** 0.5  # RMSE

        # Запускаем Optuna
        study = optuna.create_study(direction="minimize", study_name=task)
        study.optimize(objective, n_trials=10, n_jobs=-1)
        return study


    def train_on_partitions(self, df: dd.DataFrame, study_params: dict[str, int | float], target) -> CatBoostRegressor:
        """Обучаем модель по партициям данных"""
        params = {
                "learning_rate": study_params.best_params.get("learning_rate"),
                "l2_leaf_reg": study_params.best_params.get("l2_leaf_reg"),
                "max_depth": study_params.best_params.get("max_depth"),
                "iterations": study_params.best_params.get("iterations"),
        }
        model = CatBoostRegressor(**params, verbose=50)

        # Проходим по чанкам и обучаем модель по частям
        partitions = df.to_delayed()
        for i, partition in enumerate(partitions):
            model = self.training_pipe(model, partition, i=i, target=target)

        return model

    def save_model(self, model, model_path):
        """Сохраняем модель в файл. В функцию подается возвращаемый инстанс модели из функции train_on_partitions"""
        joblib.dump(model, model_path)

    def predict_on_test_partitions(self, df: dd.DataFrame, model, target):
        """Делаем прогнозы"""
        
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
    
    def compute_model_metrics(self, val_true, val_preds):
        """Считаем метрики для модели"""
        mse = mean_squared_error(val_true, val_preds)
        rmse = (np.sqrt(mean_squared_error(val_true, val_preds)))
        mae = mean_absolute_error(val_true, val_preds)
        mape = mean_absolute_percentage_error(val_true, val_preds)
        r2 = r2_score(val_true, val_preds)
        print("Оценка тестового множества модели CatBoost:")
        print('MSE: {:.2f}'.format(mse))
        print('RMSE: {:.2f}'.format(rmse))
        print('MAE: {:.2f}'.format(mae))
        print('MAPE: {:.2f}'.format(mape))
        print('R2: {:.2f}'.format(r2))
