import dask.dataframe as dd
import yaml
import os

class DataLoader:
    def __init__(self, filepath: str, config_path: str):
        self.filepath = os.path.join(os.path.dirname(__file__), '..', 'dataset', filepath)
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'config', config_path)
        self.columns = self._load_config()

    def _load_config(self):
        """Загружает список колонок из YAML-конфига"""
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        columns = config.get("columns", [])
        return columns

    def load(self):
        """Загружает CSV в Dask DataFrame, используя колонки из конфига"""
        data_dask = dd.read_csv(self.filepath, assume_missing=True)
        data_dask = data_dask[self.columns]
        return data_dask
