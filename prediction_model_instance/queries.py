# Блок с запросами к PostgreSQL DB

def get_query(
    task: str
) -> str:
    """Метод для получения нужного запроса к базе данных
    """
    queries_dict = {"car_sales_pred": CAR_SALES_COUNT_QUERY,
    }
    query_str =  queries_dict.get(task, "")
    return query_str.format(task)

# Запрос к таблице car_sales
CAR_SALES_COUNT_QUERY = """SELECT COUNT(*) FROM car_sales"""