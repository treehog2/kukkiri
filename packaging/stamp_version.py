import re
import sys
from pathlib import Path

version = sys.argv[1]
for path, pattern in (("pyproject.toml", r'^version = ".*"$'), ("kukkiri/__init__.py", r'^__version__ = ".*"$')):
    p = Path(path)
    text, n = re.subn(pattern, pattern.split(" = ")[0].lstrip("^") + f' = "{version}"', p.read_text(), count=1,
                      flags=re.M)
    if n != 1:
        sys.exit(f"no version line in {path}")
    p.write_text(text)
print("version", version)
