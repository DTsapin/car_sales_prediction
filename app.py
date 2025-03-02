from fastapi import FastAPI
import uvicorn
from prediction_model_instance.load_model import CarSaleFactors, ModelLoader
import pandas as pd
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Загрузка модели при старте сервиса
model_loader = ModelLoader('model.pkl')

@asynccontextmanager
async def lifespan(app: FastAPI):
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

    data_for_pred = pd.DataFrame([cars.model_dump()])
 
    prediction = np.exp(model.predict(data_for_pred)) - 1

    return round(float(prediction[0]), 1)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4444)