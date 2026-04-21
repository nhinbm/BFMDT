import gdown
import os
import sys
import urllib.request
import zipfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DATA_DIR,
    GDRIVE_FOLDER_ID,
    YALE_DIR,
    YALE_MAT_PATH,
    YALE_MAT_URL,
)


def download_datasets():
    """
    Download datasets from Google Drive folder to local datasets/data/ directory.
    The folder now contains zip files and CSV, which will be downloaded and unzipped automatically.
    Also fetches Yale.mat from scikit-feature (the paper's actual Yale source).
    """
    output_dir = DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    gdown.download_folder(id=GDRIVE_FOLDER_ID, output=output_dir, quiet=False)

    nested_dir = os.path.join(output_dir, 'data')
    if os.path.isdir(nested_dir):
        for item in os.listdir(nested_dir):
            src = os.path.join(nested_dir, item)
            dst = os.path.join(output_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(nested_dir)

    for file in os.listdir(output_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(output_dir, file)
            extract_dir = os.path.join(output_dir, file[:-4])
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            inner_dir = os.path.join(extract_dir, file[:-4])
            if os.path.exists(inner_dir) and os.path.isdir(inner_dir):
                for item in os.listdir(inner_dir):
                    shutil.move(os.path.join(inner_dir, item), extract_dir)
                os.rmdir(inner_dir)

            os.remove(zip_path)

    os.makedirs(YALE_DIR, exist_ok=True)
    if not os.path.exists(YALE_MAT_PATH):
        print(f"Downloading Yale.mat from {YALE_MAT_URL} ...")
        urllib.request.urlretrieve(YALE_MAT_URL, YALE_MAT_PATH)


if __name__ == '__main__':
    download_datasets()