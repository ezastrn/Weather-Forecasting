import site
from pathlib import Path

site.addsitedir(r"C:\Assesment\.venv\Lib\site-packages")
import nbformat

path = Path(r"C:\Assesment\Faeyza Safa Izz Deyardi-255150207111060-ml.ipynb")
nb = nbformat.read(path, as_version=4)

# Cell 2: Setup markdown - Colab instructions only.
nb.cells[2].source = """## Setup

Memuat library, membaca data, memparse timestamp, serta menerapkan clipping anomali fisik sebelum EDA dan feature engineering. Model yang diimpor hanya Linear Regression, Ridge, Lasso, dan Elastic Net.

### Menjalankan di Google Colab

Cell Setup memuat `weather.csv` dengan urutan berikut:

1. **Working directory:** jika `weather.csv` sudah berada di folder kerja saat ini (baik lokal maupun sesi Colab), file itu langsung digunakan.
2. **Google Drive (opsional, hanya Colab):** jika file belum ada, cell akan me-mount Google Drive dan mencari file pada `DRIVE_DATASET_PATH`. Ubah placeholder variabel tersebut sesuai folder Drive Anda sebelum menjalankan cell.
3. **Upload manual (fallback terakhir, hanya Colab):** jika file tidak ditemukan di Drive, Colab akan menampilkan pemilih file untuk upload CSV.

Untuk VS Code/lokal, letakkan `weather.csv` di folder kerja yang sama dengan notebook. Tidak ada path desktop atau path Google Drive pribadi yang disimpan di notebook."""

# Cell 3: Setup code. Everything after data loading is preserved byte-for-byte.
old = nb.cells[3].source
start = old.index("import warnings")
marker = "df_raw['timestamp'] = pd.to_datetime"
after = old[old.index(marker):]

new_prefix = """import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Environment and dataset configuration. Keep this placeholder relative to Google Drive;
# edit it only when you want to load a persistent copy from your own Drive.
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DATA_FILE = Path('weather.csv')
DRIVE_DATASET_PATH = 'MyDrive/ML-Assessment/weather.csv'  # Change to your own Drive-relative dataset path when needed.
method_used = None

# 1) Use a session/local working-directory file whenever it already exists.
if DATA_FILE.exists():
    method_used = 'local working directory'

# 2) In Colab only, optionally mount Drive and look for the configured relative path.
elif IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    drive_file = Path('/content/drive') / DRIVE_DATASET_PATH
    if drive_file.exists():
        DATA_FILE = drive_file
        method_used = 'Google Drive mount'

    # 3) Last fallback for Colab: interactive upload.
    else:
        from google.colab import files
        print('Dataset was not found in the working directory or configured Google Drive location. Please upload the CSV file.')
        uploaded = files.upload()
        csv_files = [Path(name) for name in uploaded if name.lower().endswith('.csv')]
        if Path('weather.csv') in csv_files:
            DATA_FILE = Path('weather.csv')
        elif len(csv_files) == 1:
            DATA_FILE = csv_files[0]
        else:
            raise FileNotFoundError('Upload exactly one CSV file, preferably named weather.csv, then re-run this cell.')
        method_used = 'manual upload'

# Local execution has no Drive/upload fallback by design.
else:
    raise FileNotFoundError('weather.csv was not found. Letakkan weather.csv di folder yang sama dengan notebook ini.')

print(f'Dataset dimuat via: {method_used} -> {DATA_FILE.name}')
TARGET = 'target_temperature_6h_c'
df_raw = pd.read_csv(DATA_FILE)
"""

nb.cells[3].source = new_prefix + after
nbformat.write(nb, path)
print(f'Updated cells 2 and 3 only: {path.name}')
