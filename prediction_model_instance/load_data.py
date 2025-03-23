import dask.dataframe as dd
from sqlalchemy import create_engine, text
from prediction_model_instance.queries import get_query
import os

class DataLoader:
    
    def __init__(self):
        self.db_connection_string = os.getenv("DB_CONNECT_STRING")
    
    def create_connection(self,) -> None:
        """Метод для осуществления подключения к базе данных"""

        alchemyEngine = create_engine(self.db_connection_string, pool_recycle=3600,
            )
        postgreSQLConnection = alchemyEngine.connect()
        return postgreSQLConnection
    
    def compute_number_of_partitions(self, task):
        # Создаем соединение с базой данных
        conn = self.create_connection()

        # Получаем общее количество строк в таблице
        query = get_query(task)
        num_rows = conn.execute(text(query)).scalar()

        # Рассчитываем количество партиций
        chunk_size = 15_000
        n_partitions = max(1, num_rows // chunk_size)
        return n_partitions

    def get_data(self, table_name: str, n_partitions: int) -> dd.DataFrame:
        """Получение данных из базы PostgreSQL
        """

        data_dask = dd.read_sql(table_name, self.db_connection_string, npartitions = n_partitions, index_col="id")
        
        return data_dask
