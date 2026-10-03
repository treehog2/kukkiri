import os
import sys
from pathlib import Path

app = Path(sys.executable).resolve().parent
os.environ["PATH"] = str(app / "ffmpeg") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("KUKKIRI_WORKER_PYTHON", str(app / "python.exe"))

from kukkiri.gui import main

main()
