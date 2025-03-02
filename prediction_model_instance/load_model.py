import joblib
import os
from prediction_model_instance.train_model import ModelTrainer
from prediction_model_instance.load_data import DataLoader
from pydantic import BaseModel
import os

# Инициализация класса, описывающего входные факторы модели
class CarSaleFactors(BaseModel):
    brand: str
    year: float
    mileage: float
    body_type: str
    fuel_type: str
    engine_volume: float
    engine_power: float
    transmission: str
    drive: str
    wheel: str
    state: str
    owners_count: float
    pts: str
    customs: str

class ModelLoader:
    def __init__(self, model_path: str, data_path: str):
        self.model_path = os.path.join(os.path.dirname(__file__), 'model_object', model_path)
        self.data_loader = DataLoader(filepath=data_path, config_path='config.yaml')
        self.model_trainer = ModelTrainer(self.model_path)

    def load_model(self):
        """Загружает модель из файла или обучает новую"""

        if os.path.exists(self.model_path):
            print("Модель найдена! Загружаем...")
            model = joblib.load(self.model_path)
        else:
            # TODO нужна проверка на наличие директории
            print("Файл модели не найден. Запускаем подготовку данных и обучение...")
            df = self.data_loader.load()
            model = self.model_trainer.train_on_partitions(df)
            joblib.dump(model, self.model_path)
            print("Модель обучена и сохранена!")

        return model
