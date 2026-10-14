import site
from pathlib import Path

site.addsitedir(r"C:\Assesment\.venv\Lib\site-packages")

import nbformat
from nbclient import NotebookClient

notebook_path = Path("nama-nim-ml.ipynb")
notebook = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(
    notebook,
    kernel_name="local_weather_kernel",
    timeout=1200,
    resources={"metadata": {"path": str(Path.cwd())}},
)
client.execute()
nbformat.write(notebook, notebook_path)
print(f"Executed and saved: {notebook_path}")
