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
    def __init__(self, model_name: str, model_directory: str):
        self.model_directory = model_directory
        self.model_path = os.path.join(os.path.dirname(__file__), model_directory, model_name)
    
    def check_model_instance_dir(self) -> None:
        """
        Функция для автоматической проверки и создания директории с инстансом модели (если она отсутствует).
        """
        directory_path = os.path.join(os.path.dirname(__file__), self.model_directory)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def load_model(self):
        """Загружает модель из файла или обучает новую"""

        if os.path.exists(self.model_path):
            print("Модель найдена! Загружаем...")
            model = joblib.load(self.model_path)

        else:
            print("Файл модели не найден. Запускаем подготовку данных и обучение...")
            self.check_model_instance_dir()
            data_loader = DataLoader()
            # TODO использовать таски для разных таблиц данных
            df = data_loader.get_data('car_sales')
            model_trainer = ModelTrainer(self.model_directory)
            # На будущее - куда-то надо логировать прогнозы модели на тестовом датасете
            train_df, test_df = model_trainer.split_data(df)
            model = model_trainer.train_on_partitions(train_df)
            # Сохраняем модель
            model_trainer.save_model(model, self.model_path)
            print("Модель обучена и сохранена!")

        return model
