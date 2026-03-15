import gdown
import os
import zipfile
import shutil

def download_datasets():
    """
    Download datasets from Google Drive folder to local datasets/data/ directory.
    The folder now contains zip files and CSV, which will be downloaded and unzipped automatically.
    """
    folder_id = '1fnjGJ5VBu_m7XSfieKE3LU9prTb6hyjk'
    output_dir = 'datasets/data/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading datasets to {output_dir}...")
    gdown.download_folder(id=folder_id, output=output_dir, quiet=False)
    
    # Unzip any .zip files
    for file in os.listdir(output_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(output_dir, file)
            extract_dir = os.path.join(output_dir, file[:-4])  # Remove .zip extension
            print(f"Unzipping {file} to {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Check if there's a nested folder with the same name and flatten it
            inner_dir = os.path.join(extract_dir, file[:-4])
            if os.path.exists(inner_dir) and os.path.isdir(inner_dir):
                print(f"Flattening nested folder {inner_dir}...")
                for item in os.listdir(inner_dir):
                    shutil.move(os.path.join(inner_dir, item), extract_dir)
                os.rmdir(inner_dir)
            
            os.remove(zip_path)  # Remove the zip file after extraction
            print(f"Removed {file} after extraction.")
    
    print("Download and extraction completed.")

if __name__ == '__main__':
    download_datasets()