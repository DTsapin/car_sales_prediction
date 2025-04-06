# Используем слим-образ питона
FROM python:3.11.9-slim

# Указываем рабочий каталог
WORKDIR /car_sales_prediction

# Доставляем системные зависимости
RUN apt-get update && apt install -y nano git curl wget libpq-dev gcc

# Устанавливаем poetry
RUN pip install poetry

# Копируем зависимости и устанавливаем их
COPY pyproject.toml poetry.lock /tmp/
WORKDIR /tmp
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Копируем исходный код
WORKDIR /car_sales_prediction
COPY . .

ENV PYTHONPATH=.

# Запуск приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
