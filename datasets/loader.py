import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine, fetch_openml
from sklearn.preprocessing import LabelEncoder

class DatasetRegistry:
    """
    Registry pattern to manage and load benchmark datasets.
    
    Args:
        name (str): Unique name of the dataset to register or load.
        
    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision labels of shape (n_samples,).
    """
    
    _registry = {}

    @classmethod
    def register(cls, name):
        """Decorator to register a dataset loader function under a given name."""
        def wrapper(func):
            cls._registry[name] = func
            return func
        return wrapper

    @classmethod
    def get_loader(cls, name):
        if name not in cls._registry:
            raise KeyError(f"Dataset '{name}' has not been registered in the Registry.")
        return cls._registry[name]

    @classmethod
    def list_datasets(cls):
        return list(cls._registry.keys())

def fetch_or_load_local(name, openml_name=None , data_dir="datasets/data/"):
    """
    Try loading from OpenML; if the network connection is lost, read the local CSV file. 
    If there is no OpenML name, read the local file directly.
    
    Args:
        name (str): Unique name of the dataset to load.
        openml_name (str): The corresponding OpenML dataset name to attempt fetching from. If None, skip OpenML.
        
    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision labels of shape (n_samples,).
    """
    X, y_raw = None, None
    
    if openml_name:
        try:
            print(f"[Loader] Fetching dataset '{name}' from OpenML...")
            if name == 'SMK_CAN_187':
                data = fetch_openml(name=openml_name, version=1, as_frame=True, parser='auto')
                X = data.data.values
                y_raw = data.target.values
            else:
                data = fetch_openml(name=openml_name, version=1, as_frame=False, parser='auto')
                X, y_raw = data.data, data.target
            print(f" -> API download successful!")
        except Exception as e:
            print(f" -> Network/API Error: {e}. Switch to reading local files...")
            openml_name = None
            
    if not openml_name:
        file_path = f"{data_dir}{OPENML_DATASETS[name]}.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'. Please download the data manually!")
        print(f"[Loader] Reading from {file_path}...")
        df = pd.read_csv(file_path)
        if name == 'turkiye-student':
            X = df.drop(df.columns[4], axis=1).values
            y_raw = df.iloc[:, 4].values
        else:
            X = df.iloc[:, :-1].values
            y_raw = df.iloc[:, -1].values

    y = LabelEncoder().fit_transform(y_raw)
    return X, y_raw

OPENML_DATASETS = {
    'breast-wisconsin': 'breast-w',
    'breast-cancer': 'breast-cancer',
    'heart-disease': 'heart-statlog',
    'hepatitis': 'hepatitis',
    'german-credit': 'credit-g',
    'vehicle': 'vehicle',
    'diabetes': 'diabetes',
    'wine-quality': 'wine-quality-white',
    'sonar': 'sonar',
    'arcene': 'arcene',
    'turkiye-student': 'turkiye-student-evaluation',
    # 'DrivFace': 'DrivFace',
    'wdbc': 'wdbc',
    'divorce': 'divorce_prediction',
    # 'Yale': 'Yale',
    'SMK_CAN_187': 'SMK',
    # 'PEMS-SF': 'PEMS-SF'
}

# TODO: Add more datasets {DrivFace, Yale, PEMS-SF}.

for ds_name, openml_name in OPENML_DATASETS.items():
    @DatasetRegistry.register(ds_name)
    def _loader(n=ds_name, on=openml_name):
        return fetch_or_load_local(n, on)

@DatasetRegistry.register('wine')
def get_wine_data():
    data = load_wine()
    return data.data, data.target

def load_dataset(name, data_dir="datasets/data/"):
    """
    Load a dataset by name.
    If it's registered in the DatasetRegistry, use the registered loader.
    Otherwise, attempt to load from OpenML or local CSV.
    
    Args:
        name (str): Unique name of the dataset to load.
        data_dir (str): Directory to look for local CSV files if OpenML loading fails.
        
    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision labels of shape (n_samples,).
    """
    try:
        loader_func = DatasetRegistry.get_loader(name)
        return loader_func()
    except KeyError:
        return fetch_or_load_local(name, openml_name=None, data_dir=data_dir)

def list_available_datasets():
    return DatasetRegistry.list_datasets()