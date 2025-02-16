# Импорт библиотек
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



# Инициализация класса, описывающего входные факторы модели
class CarSaleFactors(BaseModel):
    year: float 
    km_driven: float 
    fuel: str
    seller_type: str
    transmission: str
    owner: str

# Инициализация класса обучения и прогнозов модели
class CarSaleModel:
    # Конструктор класса, загружаем датасет и стейт обученной модели,
    # если он имеется. Если нет - вызываем функцию _train_model и только после этого сохраняется стейт модели
    def __init__(self):
        self.df = pd.read_csv('prediction_model_instance/CAR DETAILS FROM CAR DEKHO.csv')
        self.model_fname_ = 'prediction_model_instance/car_sale_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except FileNotFoundError as _:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)

    def _preproc_data():
        pass

    def _create_fetures():
        pass

    # Непосредственно, функция подготовки данных и обучения модели CatBoostRegressor, возвращает экземпляр модели
    def _train_model(self):
        self.df = self.df.drop(['name'], axis=1)
        target = self.df['selling_price']
        features = self.df.drop(['selling_price'], axis=1)
    
        # Явно определяем списки категориальных признаков
        cat_features = [col for col in features.columns if features[col].dtype != 'int64']

        # Преобразуем категориальные признаки в тип 'category'
        features[cat_features] = features[cat_features].apply(lambda x: x.astype('category'))

        # Инициализация модели CatBoost
        catboost_model = CatBoostRegressor(cat_features=cat_features, n_estimators=100)

        # Обучение модели
        model = catboost_model.fit(features, target)
    
        return model

    # Непосредственно, функция прогнозов на основе входного массива факторов, возвращает значение прогноза
    def predict_sales(self, diction):
        prediction = self.model.predict(diction)
        return f"Прогнозная стоимость автомобиля: {prediction[0]}"