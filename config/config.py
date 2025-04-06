# Таски для обучения моделей


# Названия SQL таблиц для тасков
task_tables_dict = {
    "car_sales_pred": "car_sales",
}

# Колонки таргетов для тасков
targets_dict = {
    "car_sales_pred": "price",
}

# Словарь для связи эндпоинтов с тасками
models_mapping_dict = {
    "get_sale_prediction_for_one_car": "car_sales_pred",
    "get_sale_prediction_for_multiple_cars": "car_sales_pred",
}