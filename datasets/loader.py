import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.datasets import load_wine, fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from typing import Tuple, Optional

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

def fetch_or_load_local(name, openml_name=None , data_dir="datasets/data/")-> Tuple[np.ndarray, np.ndarray]:
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
                X = np.asarray(data.data)
                y_raw = np.asarray(data.target)
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
    return X, y

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
    'SMK_CAN_187': 'SMK',
    # 'PEMS-SF': 'PEMS-SF'
}

# TODO: Add more datasets {PEMS-SF}.
# DrivFace is handled by a custom image loader below rather than OpenML.

for ds_name, openml_name in OPENML_DATASETS.items():
    @DatasetRegistry.register(ds_name)
    def _loader(n=ds_name, on=openml_name):
        return fetch_or_load_local(n, on)

@DatasetRegistry.register('wine')
def get_wine_data():
    data = load_wine()
    return data.data, data.target

def load_drivface_data(data_dir="datasets/data/DrivFace/", image_size=(80,80)):
    """
    Load the DrivFace head-pose dataset and convert images to fixed-size feature vectors.
    The original paper "Fusing Monotonic Decision Tree Based on Related Family" reports
    606 objects with 6 400 features and 3 decision classes.  We replicate that by
    reading the ground truth file ``drivPoints.txt`` and resizing/cropping each
    face image to ``image_size`` (default 80×80 = 6 400 pixels).

    Args:
        data_dir (str): Base directory containing ``drivPoints.txt`` and ``DrivImages``.
        image_size (tuple): (width, height) to resize the face crop to.

    Returns:
        X (np.ndarray): Feature matrix shaped (n_samples, image_size[0]*image_size[1]).
        y (np.ndarray): Integer labels 1/2/3 corresponding to left/frontal/right poses.
    """
    points_file = os.path.join(data_dir, "drivPoints.txt")
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"DrivFace points file not found at '{points_file}'")

    df = pd.read_csv(points_file)
    images = []
    labels = []

    img_dir = os.path.join(data_dir, "DrivImages")
    for _, row in df.iterrows():
        fname_base = row["fileName"].strip()
        # the actual files in the repository include a trailing space before ".jpg";
        # locate the file by matching the prefix instead of assuming an exact name.
        candidates = [f for f in os.listdir(img_dir) if f.startswith(fname_base)]
        if not candidates:
            raise FileNotFoundError(f"No image file starting with '{fname_base}' in {img_dir}")
        img_path = os.path.join(img_dir, candidates[0])

        try:
            img = Image.open(img_path).convert('L')  # grayscale
            # crop to face bounding box if available
            if all(col in row for col in ["xF", "yF", "wF", "hF"]):
                x, y, w, h = int(row["xF"]), int(row["yF"]), int(row["wF"]), int(row["hF"])
                box = (x, y, x + w, y + h)
                img = img.crop(box)
            img = img.resize(image_size)
            arr = np.array(img).flatten()
            images.append(arr)
            labels.append(int(row["label"]))
        except Exception as ex:
            print(f"Warning: failed to process {img_path}: {ex}")
            continue

    if not images:
        raise ValueError(f"No images could be loaded from '{img_dir}'")

    X = np.vstack(images)
    y = np.array(labels)
    print(f"[Loader] DrivFace loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y


def load_yale_data(data_dir="datasets/data/yale/", n_components=1025):
    """
    Load Yale Face Database images and convert to feature vectors.
    Yale images are stored as GIF files without extensions.
    Applies dimensionality reduction as per paper: 
    "Fusing Monotonic Decision Tree Based on Related Family"
    
    Args:
        data_dir (str): Path to Yale dataset directory
        n_components (int): Number of components to keep (default: 1025)
                          Note: Will be capped at min(n_samples, n_features) for PCA
    
    Returns:
        X (np.ndarray): Dimensionality-reduced feature matrix, shape (n_samples, n_components).
        y (np.ndarray): Labels representing subject IDs (0-14 for 15 subjects).
    """
    yale_path = data_dir
    if not os.path.exists(yale_path):
        raise FileNotFoundError(f"Yale dataset directory not found at '{yale_path}'")
    
    images = []
    labels = []
    
    # Get all files in the directory (excluding README and directories)
    all_files = [f for f in os.listdir(yale_path) 
                 if os.path.isfile(os.path.join(yale_path, f)) 
                 and f.lower() != 'readme.txt']
    
    # Sort files for consistent ordering
    all_files.sort()
    
    if not all_files:
        raise FileNotFoundError(f"No image files found in '{yale_path}'")
    
    # Load images and extract subject labels
    for filename in all_files:
        file_path = os.path.join(yale_path, filename)
        try:
            # Open image and convert to grayscale
            img = Image.open(file_path).convert('L')
            # Flatten image to 1D array
            img_array = np.array(img).flatten()
            images.append(img_array)
            
            # Extract subject number from filename (e.g., 'subject01.normal' -> 0)
            subject_id = int(filename.split('.')[0].replace('subject', '')) - 1
            labels.append(subject_id)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
    
    if not images:
        raise ValueError(f"Failed to load any images from '{yale_path}'")
    
    X_raw = np.array(images)
    y = np.array(labels)
    
    n_samples = X_raw.shape[0]
    n_features = X_raw.shape[1]
    
    # PCA can extract at most min(n_samples, n_features) components
    # For Yale (165 samples), max is 165 components using PCA
    # Use Gaussian Random Projection for arbitrary target dimensions
    if n_components >= n_samples:
        print(f"[Loader] Using Gaussian Random Projection: {n_features} -> {n_components} features")
        transformer = GaussianRandomProjection(n_components=n_components, random_state=42)
        X = transformer.fit_transform(X_raw)
    else:
        print(f"[Loader] Using PCA: {n_features} -> {n_components} features")
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(X_raw)
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"[Loader] PCA explained variance ratio: {explained_variance:.4f}")
    
    print(f"[Loader] Yale dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    return X, y

def load_pems_sf_data(data_dir="datasets/data/pems-sf/"):
    """
    Load the PEMS-SF dataset from raw UCI files.
    Expected data: 440 objects, 138672 features, 7 classes.
    Automatically caches to .npy for ultra-fast loading on subsequent runs.
    """
    pems_path = data_dir
    if not os.path.exists(pems_path):
        raise FileNotFoundError(f"PEMS-SF directory not found at '{pems_path}'")
    
    npy_x_path = os.path.join(pems_path, "X_pems.npy")
    npy_y_path = os.path.join(pems_path, "y_pems.npy")
    
    if os.path.exists(npy_x_path) and os.path.exists(npy_y_path):
        print("[Loader] Loading PEMS-SF from ultra-fast cache (.npy)...")
        X = np.load(npy_x_path)
        y_raw = np.load(npy_y_path)
    else:
        print("[Loader] Cache not found, starting to parse raw PEMS-SF data (first time only)...")
        
        def _parse_features(filename):
            filepath = os.path.join(pems_path, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing file: {filepath}")
            print(f" -> Parsing {filename}...")
            
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    cleaned = line.replace('[', '').replace(']', '').replace(';', ' ').replace(',', ' ')
                    row = np.fromstring(cleaned, dtype=float, sep=' ')
                    data.append(row)
            return np.array(data)

        def _parse_labels(filename):
            filepath = os.path.join(pems_path, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing file: {filepath}")
            
            with open(filepath, 'r') as f:
                content = f.read().replace('[', '').replace(']', '').replace(';', ' ').replace(',', ' ')
                labels = np.fromstring(content, dtype=int, sep=' ')
            return labels

        X_train = _parse_features("PEMS_train")
        y_train = _parse_labels("PEMS_trainlabels")
        
        X_test = _parse_features("PEMS_test")
        y_test = _parse_labels("PEMS_testlabels")

        X = np.vstack((X_train, X_test))
        y_raw = np.concatenate((y_train, y_test))

        print(" -> Saving cache (.npy) to speed up future runs...")
        np.save(npy_x_path, X)
        np.save(npy_y_path, y_raw)

    n_samples, n_features = X.shape
    classes = len(np.unique(y_raw))
    print(f"[Loader] PEMS-SF successfully loaded: {n_samples} samples, {n_features} features, {classes} classes")
    
    if n_samples != 440 or n_features != 138672:
        print(f" -> [WARNING] Dimensions do not match the paper! (Expected: 440x138672, Current: {n_samples}x{n_features})")
    
    y = LabelEncoder().fit_transform(y_raw)
    
    return X, y

@DatasetRegistry.register('DrivFace')
def get_drivface_data():
    """Convenience wrapper for the DrivFace loader."""
    return load_drivface_data()


@DatasetRegistry.register('Yale')
def get_yale_data():
    return load_yale_data()

@DatasetRegistry.register('PEMS-SF')
def get_pems_sf_data():
    return load_pems_sf_data()

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