import site
from pathlib import Path

site.addsitedir(r"C:\Assesment\.venv\Lib\site-packages")
import nbformat
from nbclient import NotebookClient

path = Path(r"C:\Assesment\Faeyza Safa Izz Deyardi-255150207111060-ml.ipynb")
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    kernel_name="local_weather_kernel",
    timeout=1200,
    resources={"metadata": {"path": r"C:\Assesment"}},
)
client.execute()
nbformat.write(notebook, path)
print(f"Executed successfully: {path.name}")
