from fastapi import FastAPI, File, UploadFile
import uvicorn
from prediction_model_instance.load_model import CarSaleFactors, ModelLoader
import pandas as pd
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse
from config.config import targets_dict, models_mapping_dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI-функция жизненного цикла сервиса"""
    task_names = list(targets_dict.keys())
    # Загрузка модели при старте сервиса TODO избавиться тут от хардкода, разнести модели по таскам
    model_loader = ModelLoader()
    # Загружаем словарь с моделями
    models = model_loader.load_model(task_names)
    # Сохраняем модели в app.state
    app.state.models = models
    yield
    print("Приложение завершает свою работу.")


app = FastAPI(lifespan=lifespan)

# Разрешаем запросы с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/get_sale_prediction_for_one_car')
async def get_sale_prediction_for_one_car(car: CarSaleFactors):
    """
        FastAPI-эндпоинт для прогноза стоимости авто.

        Args
        ----------
        car: Class, pydantic-модель, описывающая факторы стоимости авто

        Returns
        -------
        prediction: float, прогнозная стоимость авто
    """
    
    model_name = models_mapping_dict['get_sale_prediction_for_one_car']
    model = app.state.models[model_name]

    # Делаем датафрейм, где колонки - поля pydantic-класса
    data_for_pred = pd.DataFrame([car.model_dump()])

    # Прогноз с обратным логарифмированием
    prediction = np.exp(model.predict(data_for_pred)) - 1

    return round(float(prediction[0]), 1)

@app.post("/get_sale_prediction_for_multiple_cars")
async def get_sale_prediction_for_multiple_cars(file: UploadFile = File(...)):

    """
        FastAPI-эндпоинт для прогноза стоимости сразу нескольких авто (из excel-файла).

        Args
        ----------
        file: UploadFile = File(...), объект excel-файла

        Returns
        -------
        объект StreamingResponse, возвращающий excel-файл со значениями прогнозов
    """

    model_name = models_mapping_dict['get_sale_prediction_for_multiple_cars']
    model = app.state.models[model_name]

    # Чтение содержимого файла как байтов
    contents = await file.read()
    # Создаём объект BytesIO
    data = BytesIO(contents)

    df = pd.read_excel(data)

    # Осуществление прогноза
    df['predict_value'] = np.exp(model.predict(df)) - 1

    # Обновляем xlsx-файл
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    # Возвращаем StreamingResponse для передачи файла на лету, без записи в файловую систему сервера
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                             headers={"Content-Disposition": "attachment;filename=predictions.xlsx"})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4444)