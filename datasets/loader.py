import os
import pandas as pd
import numpy as np
from PIL import Image
from scipy.io import loadmat
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

from config import (
    DATA_DIR,
    DATASETS,
    DRIVFACE_DIR,
    DRIVFACE_IMAGE_SIZE,
    NAME_TO_ID,
    OPENML_CACHE_DIR,
    PEMS_SF_DIR,
    PEMS_SF_EXPECTED_SHAPE,
    YALE_DIR,
)


def _encode_labels(name, y_raw):
    """Encode nominal labels using an explicit mapping if defined for the dataset,
    otherwise fall back to alphabetical LabelEncoder."""
    mapping = DATASETS[name].label_mapping if name in DATASETS else None
    if mapping is not None:
        y_str = pd.Series(y_raw).astype(str)
        return y_str.map(mapping).to_numpy(dtype=np.int64)
    return LabelEncoder().fit_transform(y_raw)


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

    @classmethod
    def has(cls, name):
        return name in cls._registry


def fetch_or_load_local(name, openml_name=None, data_dir=DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
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
            if name == 'SMK_CAN_187':
                data = fetch_openml(name=openml_name, version=1, as_frame=True, parser='auto', data_home=OPENML_CACHE_DIR)
                X = np.asarray(data.data)
                y_raw = np.asarray(data.target)
            else:
                data = fetch_openml(name=openml_name, version=1, as_frame=False, parser='auto', data_home=OPENML_CACHE_DIR)
                X, y_raw = data.data, data.target
        except Exception as e:
            print(f"[Loader] OpenML fetch failed for '{name}': {e}. Falling back to local CSV.")
            openml_name = None

    if not openml_name:
        file_path = os.path.join(data_dir, f"{NAME_TO_ID[name]:02d}_{name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'. Please download the data manually!")
        df = pd.read_csv(file_path)
        if name == 'turkiye-student':
            X = df.drop(df.columns[4], axis=1).values
            y_raw = df.iloc[:, 4].values
        else:
            X = df.iloc[:, :-1].values
            y_raw = df.iloc[:, -1].values

    if name in DATASETS and DATASETS[name].allow_missing_values:
        X_df = pd.DataFrame(X)
        y_s = pd.Series(y_raw)
        mask = ~(X_df.isin(['?']).any(axis=1) | X_df.isna().any(axis=1) | y_s.isna())
        X, y_raw = X_df[mask].values, y_s[mask].values

    y = _encode_labels(name, y_raw)
    return X, y


def load_drivface_data(data_dir=DRIVFACE_DIR, image_size=DRIVFACE_IMAGE_SIZE):
    """
    Load the DrivFace head-pose dataset and convert images to fixed-size feature vectors.
    The original paper "Fusing Monotonic Decision Tree Based on Related Family" reports
    606 objects with 6 400 features and 3 decision classes.  We replicate that by
    reading the ground truth file ``drivPoints.txt`` and resizing/cropping each
    face image to ``image_size`` (default 80×80 = 6 400 pixels).

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
    all_imgs = os.listdir(img_dir)

    for _, row in df.iterrows():
        fname_base = row["fileName"].strip()
        candidates = [f for f in all_imgs if f.startswith(fname_base)]
        if not candidates:
            raise FileNotFoundError(f"No image file starting with '{fname_base}' in {img_dir}")
        img_path = os.path.join(img_dir, candidates[0])

        try:
            img = Image.open(img_path).convert('L')  # grayscale
            if all(col in row for col in ["xF", "yF", "wF", "hF"]):
                bx, by, bw, bh = int(row["xF"]), int(row["yF"]), int(row["wF"]), int(row["hF"])
                img = img.crop((bx, by, bx + bw, by + bh))
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
    return X, y


def load_yale_data(data_dir=YALE_DIR):
    """
    Load the Yale Face dataset from the scikit-feature ``Yale.mat`` distribution
    (originally curated by Deng Cai), which the paper "Fusing Monotonic Decision
    Tree Based on Related Family" uses (source: jundongl in Table 4).

    The .mat file contains face-cropped images already resized to 32x32 (1024
    features per sample) under keys ``X`` (uint8 pixel intensities) and ``Y``
    (1-indexed subject ids).

    Args:
        data_dir (str): Directory containing ``Yale.mat``.

    Returns:
        X (np.ndarray): Feature matrix of shape (165, 1024).
        y (np.ndarray): Labels of shape (165,), 0-indexed for 15 subjects.
    """
    mat_path = os.path.join(data_dir, "Yale.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Yale.mat not found at '{mat_path}'. Download from "
            "https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/Yale.mat"
        )

    mat = loadmat(mat_path)
    X = np.asarray(mat['X'], dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(mat['Y']).ravel())
    return X, y


def load_pems_sf_data(data_dir=PEMS_SF_DIR):
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
        X = np.load(npy_x_path)
        y_raw = np.load(npy_y_path)
    else:
        print("[Loader] Cache not found, parsing raw PEMS-SF data (first time only)...")

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
    expected_samples, expected_features = PEMS_SF_EXPECTED_SHAPE
    if n_samples != expected_samples or n_features != expected_features:
        print(
            f"[Loader] [WARNING] PEMS-SF dimensions do not match the paper! "
            f"(Expected: {expected_samples}x{expected_features}, "
            f"Current: {n_samples}x{n_features})"
        )

    y = LabelEncoder().fit_transform(y_raw)
    return X, y

@DatasetRegistry.register('breast-cancer')
def _load_breast_cancer():
    from datasets.loader_breast_cancer import load_breast_cancer_data
    return load_breast_cancer_data()


@DatasetRegistry.register('heart-disease')
def _load_heart_disease():
    from datasets.loader_heart_disease import load_heart_disease_data
    return load_heart_disease_data()


@DatasetRegistry.register('hepatitis')
def _load_hepatitis():
    from datasets.loader_hepatitis import load_hepatitis_data
    return load_hepatitis_data()


@DatasetRegistry.register('german-credit')
def _load_german_credit():
    from datasets.loader_german_credit import load_german_credit_data
    return load_german_credit_data()


@DatasetRegistry.register('vehicle')
def _load_vehicle():
    from datasets.loader_vehicle import load_vehicle_data
    return load_vehicle_data()


@DatasetRegistry.register('diabetes')
def _load_diabetes():
    from datasets.loader_diabetes import load_diabetes_data
    return load_diabetes_data()


@DatasetRegistry.register('turkiye-student')
def _load_turkiye_student():
    from datasets.loader_turkiye_student import load_turkiye_student_data
    return load_turkiye_student_data()


@DatasetRegistry.register('DrivFace')
def _load_drivface():
    return load_drivface_data()

@DatasetRegistry.register('Yale')
def _load_yale():
    return load_yale_data()

@DatasetRegistry.register('PEMS-SF')
def _load_pems_sf():
    return load_pems_sf_data()

def load_dataset(name, data_dir=DATA_DIR):
    """
    Load a dataset by name.
    Custom-loaded datasets (image / .mat / raw text) are dispatched via
    DatasetRegistry. Everything else goes through fetch_or_load_local, which
    pulls from OpenML when openml_name is set, or falls back to the local
    {id:02d}_{name}.csv when it isn't.

    Args:
        name (str): Unique name of the dataset to load.
        data_dir (str): Directory to look for local CSV files.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision labels of shape (n_samples,).
    """
    if DatasetRegistry.has(name):
        return DatasetRegistry.get_loader(name)()
    if name in DATASETS:
        return fetch_or_load_local(name, DATASETS[name].openml_name, data_dir=data_dir)
    raise KeyError(f"Dataset '{name}' is not registered and not in DATASETS.")


def list_available_datasets():
    return list(DATASETS.keys())