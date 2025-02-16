from fastapi import FastAPI
import uvicorn
from prediction_model_instance.Model import CarSaleModel, CarSaleFactors
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
    
app = FastAPI()
model = CarSaleModel()

# Разрешаем запросы с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3333"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index():
    '''
    Первая строка.
    '''
    return {'message': 'Hello, world!'}

@app.post('/predict')
def predict_sales(cars: CarSaleFactors):
    
    data_dict = pd.DataFrame(
        {
            'year': [cars.year],
            'km_driven': [cars.km_driven],
            'fuel': [cars.fuel],
            'seller_type': [cars.seller_type],
            'transmission': [cars.transmission],
            'owner': [cars.owner],
        }
    )
 
    prediction = model.predict_sales(data_dict)
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4444)