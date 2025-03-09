import dask.dataframe as dd
import os

class DataLoader:
    
    def __init__(self):
        self.db_connection_string = os.getenv("DB_CONNECT_STRING")

    def get_data(self, table_name: str) -> dd.DataFrame:
        """Получение данных из базы PostgreSQL
        """
        # TODO брать nparitions из конфига, или же реализовать их расчет в зависимости от количества строк в таблице, 
        # прикрутить Postgre в docker
        data_dask = dd.read_sql(table_name, self.db_connection_string, npartitions = 19, index_col="id")
        
        return data_dask
