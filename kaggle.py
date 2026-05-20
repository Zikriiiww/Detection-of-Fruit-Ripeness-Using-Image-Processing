import kagglehub
from pathlib import Path
import shutil

# Download dataset
path = kagglehub.dataset_download("moltean/fruits")

# Folder sumber Tomato 11
source_folder = (
    Path(path)
    / "fruits-360_100x100"
    /"fruits-360"
    / "Test"
    / "Tomato 4"
)

# Folder tujuan milikmu sendiri
destination_folder = Path("NewDataset") / "Tomato 4"

# Buat folder jika belum ada
destination_folder.parent.mkdir(parents=True, exist_ok=True)

# Copy isi folder
shutil.copytree(
    source_folder,
    destination_folder,
    dirs_exist_ok=True
)

print("Berhasil disimpan ke:", destination_folder.resolve())