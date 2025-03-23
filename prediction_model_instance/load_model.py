import joblib
import os
from prediction_model_instance.train_model import ModelTrainer
from prediction_model_instance.load_data import DataLoader
from pydantic import BaseModel
import os
from config.config import targets_dict, task_tables_dict

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
    def __init__(self):
        self.model_directory = "model_object"
        self.model_path = os.path.join(os.path.dirname(__file__), self.model_directory)
    
    def check_model_instance_dir(self) -> None:
        """
        Функция для автоматической проверки и создания директории с инстансом модели (если она отсутствует).
        """

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def load_model(self, task_names):
        """Загружаем модель из файла или обучаем новую, если модели нет"""

        models_dict = dict.fromkeys(task_names)

        for task in task_names:
            # Получаем список файлов в директории
            files = os.listdir(self.model_path)
        
            # Отбираем файлы с расширением .pkl
            pkl_files = [f for f in files if f.endswith(f'{task}.pkl')]

            if len(pkl_files) > 0:
                print(f"Модель для задачи {task} найдена! Загружаем...")
                model_name = os.path.join(self.model_path, f"model_{task}.pkl")
                model = joblib.load(model_name)
                models_dict[task] = model

            else:
                print(f"Файл модели для задачи {task} не найден. Запускаем подготовку данных и обучение...")
                self.check_model_instance_dir()
                data_loader = DataLoader()
                # TODO использовать таски для разных таблиц данных
                n_partitions = data_loader.compute_number_of_partitions(task)
                df = data_loader.get_data(task_tables_dict[task], n_partitions)
                model_trainer = ModelTrainer()
                # Сплитим данные, оптимизируем гиперпараметры, обучаем
                train_df, test_df = model_trainer.split_data(df)
                study_params = model_trainer.opt_models(train_df, test_df, targets_dict[task], task)
                model = model_trainer.train_on_partitions(train_df, study_params, targets_dict[task])
                model_name = os.path.join(self.model_path, f"model_{task}.pkl")
                # Сохраняем модель
                model_trainer.save_model(model, model_name)
                print(f"Модель для задачи {task} обучена и сохранена!")
                models_dict[task] = model

        return models_dict
