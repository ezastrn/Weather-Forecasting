import site
import shutil
import tempfile
from pathlib import Path

site.addsitedir(r"C:\Assesment\.venv\Lib\site-packages")
import nbformat
from nbclient import NotebookClient

notebook_path = Path(r"C:\Assesment\Faeyza Safa Izz Deyardi-255150207111060-ml.ipynb")
source_csv = Path(r"C:\Assesment\weather.csv")
with tempfile.TemporaryDirectory(prefix="colab_portability_") as temp_dir:
    neutral_dir = Path(temp_dir)
    shutil.copy2(source_csv, neutral_dir / "weather.csv")
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        kernel_name="local_weather_kernel",
        timeout=1200,
        resources={"metadata": {"path": str(neutral_dir)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)

print("Executed successfully from a neutral temporary working directory.")
