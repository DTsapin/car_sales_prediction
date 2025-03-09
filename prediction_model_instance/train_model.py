import joblib
import os
import numpy as np
from catboost import CatBoostRegressor
from dask import dataframe as dd

class ModelTrainer:
    # TODO названия модели - в таск
    def __init__(self, model_directory: str, model_name: str = "model.pkl"):
        self.model_directory = model_directory
        self.model_name = model_name
        self.model_path = os.path.join(os.path.dirname(__file__), '..', model_directory, model_name)

    def split_data(self, df: dd.DataFrame):
        """Разделяем данные на обучающую и тестовую выборки"""
        train_df = df.partitions[:-1]
        test_df = df.partitions[-1]
        return train_df, test_df

    def preprocess_partition(self, batch):
        """Обрабатываем каждую партицию: удаляем пропуски, дубликаты"""
        batch = batch.dropna().drop_duplicates()
        return batch

    def train_on_partitions(self, df: dd.DataFrame) -> CatBoostRegressor:
        """Обучаем модель по партициям данных"""
        # TODO Избавиться от хардкода гиперпараметров модели, продумать
        params = {"learning_rate": 0.15, "iterations": 300, "depth": 5, "random_state": 42}
        model = CatBoostRegressor(**params, verbose=50)

        partitions = df.to_delayed()
        for i, partition in enumerate(partitions):
            partition = partition.compute()
            partition = self.preprocess_partition(partition)

            cat_features = [col for col in partition.columns if partition[col].dtype == "string[pyarrow]"]
            partition[cat_features] = partition[cat_features].astype("category")

            X_train = partition.drop(columns=["price"])
            y_train = np.log(partition["price"] + 1)

            if i == 0:
                model.fit(X_train, y_train, cat_features=cat_features)
            else:
                model.fit(X_train, y_train, cat_features=cat_features, init_model=model)

        return model

    def save_model(self, model, model_path):
        """Сохраняем модель в файл. В функцию подается возвращаемый инстанс модели из функции train_on_partitions"""
        joblib.dump(model, model_path)

    def predict_on_test_partitions(self, df: dd.DataFrame):
        """Делаем прогнозы"""
        
        predictions = []
        actuals = []
        for partition in df.to_delayed():
            batch = partition.compute()
            batch = self.preprocess_partition(batch)
            X_batch = batch.drop(columns=["price"])
            y_batch = batch["price"]
            # Обратное логарифмирование
            y_pred = np.exp(self.model.predict(X_batch)) - 1
            predictions.append(y_pred)
            actuals.append(y_batch)

        return np.concatenate(predictions), np.concatenate(actuals)
