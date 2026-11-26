import sys
from pathlib import Path


def tool(name: str) -> str:
    exe = Path(sys.executable).parent / "ffmpeg" / f"{name}.exe"
    return str(exe) if sys.platform == "win32" and exe.is_file() else name
