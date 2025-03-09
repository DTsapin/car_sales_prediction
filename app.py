from fastapi import FastAPI, File, UploadFile
import uvicorn
from prediction_model_instance.load_model import CarSaleFactors, ModelLoader
import pandas as pd
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загрузка модели при старте сервиса TODO избавиться тут от хардкода, разнести модели по таскам
    model_loader = ModelLoader('model.pkl', 'model_object')
    # Загружаем модель
    model = model_loader.load_model()
    # Сохраняем модель в app.state
    app.state.model = model
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

@app.post('/get_prediction_for_one_car')
async def get_prediction_for_one_car(cars: CarSaleFactors):
    
    model = app.state.model

    # Делаем датафрейм, где колонки - поля pydantic-класса
    data_for_pred = pd.DataFrame([cars.model_dump()])

    # Прогноз с обратным логарифмированием
    prediction = np.exp(model.predict(data_for_pred)) - 1

    return round(float(prediction[0]), 1)

@app.post("/get_prediction_for_multiple_cars")
async def get_prediction_for_multiple_cars(file: UploadFile = File(...)):

    model = app.state.model

    # Чтение содержимого файла как байтов
    contents = await file.read()
    # Создаём объект BytesIO
    data = BytesIO(contents)

    df = pd.read_excel(data)

    # Осуществление прогноза
    df['selling_price'] = np.exp(model.predict(df)) - 1

    # Обновляем xlsx-файл
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    # Возвращаем StreamingResponse для передачи файла на лету, без записи в файловую систему сервера
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                             headers={"Content-Disposition": "attachment;filename=predictions.xlsx"})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4444)