import joblib
import os
from prediction_model_instance.train_model import ModelTrainer
from prediction_model_instance.load_data import DataLoader
from pydantic import BaseModel
import os
from config.config import targets_dict, task_tables_dict

# Инициализация класса, описывающего входные факторы модели прогноза стоимости авто
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

    """
    Класс для обучения моделей.

    ...

    Атрибуты
    ----------
    model_directory: str, директория, где хранятся модели.
    model_path: str, путь до названия модели (включая название файла модели)

    Методы
    -------
    check_model_instance_dir:
        Метод для автоматической проверки и создания директории с инстансом модели (если она отсутствует).
    load_model:
        Метод для загрузки модели из файла или обучения новой, если модели нет".
    """
    def __init__(self):
        self.model_directory = "model_object"
        self.model_path = os.path.join(os.path.dirname(__file__), self.model_directory)
    
    def check_model_instance_dir(self) -> None:
        """
        Метод для автоматической проверки и создания директории с инстансом модели (если она отсутствует).
        """

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def load_model(self, task_names: list[str]):
        """
        Метод для загрузки моделей.

        Args
        ----------
        task_names: list[str], список тасков

        Returns
        -------
        models_dict: dict[CatBoostRegressor] - словарь моделей
        """

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
                n_partitions = data_loader.compute_number_of_partitions(task)
                df = data_loader.get_data(task_tables_dict[task], n_partitions)
                model_trainer = ModelTrainer()
                # Сплитим данные, оптимизируем гиперпараметры, обучаем
                train_df, test_df = model_trainer.split_data(df)
                study_params = model_trainer.opt_models(train_df, test_df, targets_dict[task], task)
                model = model_trainer.train_on_partitions(train_df, study_params, targets_dict[task])
                model_name = os.path.join(self.model_path, f"model_{task}.pkl")
                # Делаем прогнозы TODO логировать их в ClearML вместе с графиками
                val_true, val_pred = model_trainer.predict_on_test_partitions(test_df, model, targets_dict[task])
                model_trainer.compute_model_metrics(val_true, val_pred)
                # Сохраняем модель
                model_trainer.save_model(model, model_name)
                print(f"Модель для задачи {task} обучена и сохранена!")
                models_dict[task] = model

        return models_dict
