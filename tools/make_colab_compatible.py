import site
from pathlib import Path

site.addsitedir(r"C:\Assesment\.venv\Lib\site-packages")
import nbformat

path = Path(r"C:\Assesment\Faeyza Safa Izz Deyardi-255150207111060-ml.ipynb")
nb = nbformat.read(path, as_version=4)

setup_md = nb.cells[2]
setup_md.source = """## Setup

Memuat library, membaca data, memparse timestamp, serta menerapkan clipping anomali fisik sebelum EDA dan feature engineering. Model yang diimpor hanya Linear Regression, Ridge, Lasso, dan Elastic Net.

### Menjalankan di Google Colab

Unggah notebook ini ke Google Colab, lalu jalankan cell Setup. Bila `weather.csv` belum berada di direktori kerja Colab, cell akan menampilkan pemilih file untuk mengunggah dataset. Jika file sudah diunggah sebelumnya pada runtime yang sama, cell langsung menggunakannya. File harus berupa dataset `weather.csv` yang digunakan pada assessment ini; bila nama file CSV berbeda, cell akan menerima satu file CSV yang diunggah dan menggunakan file tersebut untuk sesi tersebut."""

setup_code = nb.cells[3]
old = setup_code.source
old = old.replace("import warnings\n", "import warnings\nfrom pathlib import Path\n")
old = old.replace("TARGET = 'target_temperature_6h_c'\ndf_raw = pd.read_csv('weather.csv')", """TARGET = 'target_temperature_6h_c'

# Google Colab/local compatibility: no desktop-specific path is used.
DATA_FILE = Path('weather.csv')
if not DATA_FILE.exists():
    try:
        from google.colab import files
    except ImportError as exc:
        raise FileNotFoundError(
            \"weather.csv was not found in the current folder. Place it next to this notebook, \"
            \"or run this notebook in Google Colab and upload it when prompted.\"
        ) from exc

    print('weather.csv was not found. Please upload the assessment CSV file.')
    uploaded = files.upload()
    csv_files = [Path(name) for name in uploaded if name.lower().endswith('.csv')]
    if Path('weather.csv') in csv_files:
        DATA_FILE = Path('weather.csv')
    elif len(csv_files) == 1:
        DATA_FILE = csv_files[0]
        print(f'Using uploaded CSV: {DATA_FILE.name}')
    else:
        raise FileNotFoundError(
            \"Upload exactly one CSV file, preferably named weather.csv, then re-run this cell.\"
        )

print(f'Loading data from: {DATA_FILE.resolve()}')
df_raw = pd.read_csv(DATA_FILE)""")
setup_code.source = old

nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}
nb.metadata['colab'] = {'provenance': []}
nbformat.write(nb, path)
print(f'Updated Google Colab compatibility: {path.name}')
