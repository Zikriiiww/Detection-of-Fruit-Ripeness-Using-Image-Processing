# download_dataset.py

import kagglehub

# Download dataset Fruits-360
path = kagglehub.dataset_download(
    "moltean/fruits"
)

print("Path dataset:", path)