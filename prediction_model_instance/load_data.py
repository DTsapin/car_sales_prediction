import dask.dataframe as dd
from sqlalchemy import Connection, create_engine, text
from prediction_model_instance.queries import get_query
import os

class DataLoader:
    """
    Класс для выгрузки данных из БД PostgreSQL.

    ...

    Атрибуты
    ----------
    db_connection_string : str
        Строка подключения к базе данных

    Методы
    -------
    create_connection:
        Метод для осуществления подключения к базе данных
    compute_number_of_partitions:
        Метод для расчета количества чанков с целью декомпозиции датафрейма.
    get_data:
        Метод для получения данных из БД PostgreSQL.
    """
    
    def __init__(self):
        self.db_connection_string = os.getenv("DB_CONNECT_STRING")
    
    def create_connection(self,) -> Connection:
        """
        Метод для осуществления подключения к базе данных.

        Args
        ----------
        None

        Returns
        -------
        Объект подключения к базе данных.
        """

        alchemyEngine = create_engine(self.db_connection_string, pool_recycle=3600,
            )
        postgreSQLConnection = alchemyEngine.connect()
        return postgreSQLConnection
    
    def compute_number_of_partitions(self, task: str) -> int:
        """
        Метод для декомпозиции датафрейма на чанки.

        Args
        ----------
        task: str, название таска для осуществления запроса

        Returns
        -------
        n_partitions: int, число чанков.
        """
        # Создаем соединение с базой данных
        conn = self.create_connection()

        # Получаем общее количество строк в таблице
        query = get_query(task)
        num_rows = conn.execute(text(query)).scalar()

        # Рассчитываем количество партиций
        chunk_size = int(os.getenv("N_CHUNKS"))
        n_partitions = max(1, num_rows // chunk_size)
        return n_partitions

    def get_data(self, table_name: str, n_partitions: int) -> dd.DataFrame:
        """
        Метод для получения данных из PostgreSQL.

        Args
        ----------
        table_name: str, название таблицы БД, согласно таску
        n_partitions: int, число чанков для декомпозиции датафрейма

        Returns
        -------
        data_dask: dd.Dataframe, Dask-датафрейм данных
        """

        data_dask = dd.read_sql(table_name, self.db_connection_string, npartitions = n_partitions, index_col="id")
        
        return data_dask
