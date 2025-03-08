import joblib
import os
import numpy as np
from catboost import CatBoostRegressor
from dask import dataframe as dd

class ModelTrainer:
    def __init__(self, model_path: str = "model.pkl"):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model_object', model_path)
        self.model = None

    def split_data(df: dd.DataFrame):
        """Разделяет данные на обучающую и тестовую выборки"""
        train_df = df.partitions[:-1]
        test_df = df.partitions[-1]
        return train_df, test_df

    def preprocess_partition(self, batch):
        """Обрабатываем каждую партицию: удаляем пропуски, дубликаты, логарифмируем mileage"""
        batch = batch.dropna().drop_duplicates()
        return batch

    def train_on_partitions(self, df: dd.DataFrame) -> CatBoostRegressor:
        """Обучает CatBoost по партициям"""
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

    def save_model(self):
        """Сохраняет модель в файл"""
        if self.model:
            joblib.dump(self.model, self.model_path)

    def predict_on_test_partitions(self, df: dd.DataFrame):
        """Делает прогнозы"""
        
        predictions = []
        actuals = []
        for partition in df.to_delayed():
            batch = partition.compute()
            batch = self.preprocess_partition(batch)
            X_batch = batch.drop(columns=["price"])
            y_batch = batch["price"]
            y_pred = np.exp(self.model.predict(X_batch)) - 1
            predictions.append(y_pred)
            actuals.append(y_batch)

        return np.concatenate(predictions), np.concatenate(actuals)
