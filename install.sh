#!/usr/bin/env bash
# kukkiri on Linux with an NVIDIA GPU: checks the machine, then installs kukkiri and everything it needs (kiyo-ltx
# included) into a virtual environment, ./.venv by default.
#
#   ./install.sh            the command line (kukkiri)
#   ./install.sh --gui      also the desktop app (kukkiri-gui)
#
# KUKKIRI_VENV=/path/to/venv picks another folder. Run it again to update after a `git pull`.
set -euo pipefail

here=$PWD
cd "$(dirname "$0")"
VENV=${KUKKIRI_VENV:-$PWD/.venv}
case $VENV in /*) ;; *) VENV=$here/$VENV ;; esac  # relative to where the script was run from
REQ=packaging/linux
GUI=0
for a in "$@"; do
  case $a in
    --gui) GUI=1 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

say "Checking this machine"
[ "$(uname -s)" = Linux ] || fail "this script is for Linux; on Windows use the installer, on a Mac see the README"
command -v nvidia-smi > /dev/null || fail "no NVIDIA driver found (nvidia-smi): install driver 580 or newer first"
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -1) || true
[ -n "$driver" ] && [ "${driver%%.*}" -eq "${driver%%.*}" ] 2> /dev/null \
  || fail "nvidia-smi could not read the GPU: (re)install the NVIDIA driver, 580 or newer, and reboot"
[ "${driver%%.*}" -ge 580 ] || fail "NVIDIA driver $driver is too old: kukkiri needs 580 or newer"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), driver $driver"
PY=${PYTHON:-python3}
command -v "$PY" > /dev/null || fail "python3 not found: install Python 3.12 or newer"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))' \
  || fail "$($PY -V) is too old: kukkiri needs Python 3.12 or newer (PYTHON=/path/to/python3.12 ./install.sh picks one)"
"$PY" -c 'import venv, ensurepip' 2> /dev/null \
  || fail "Python's venv module is missing: install it (Debian/Ubuntu: sudo apt install python3-venv)"
command -v ffmpeg > /dev/null || echo "warning: ffmpeg not found; kukkiri needs it on the PATH (Debian/Ubuntu: sudo apt install ffmpeg)"
if ! command -v cc > /dev/null || ! "$PY" -c 'import os, sysconfig; os.stat(sysconfig.get_paths()["include"] + "/Python.h")' 2> /dev/null; then
  echo "warning: kiyo-ltx builds GPU kernels with a C compiler and Python's headers, and one is missing"
  echo "         (Debian/Ubuntu: sudo apt install build-essential python3-dev); kiyo-v4 works without them"
fi
[ -d "$(dirname "$VENV")" ] || fail "$(dirname "$VENV") does not exist (KUKKIRI_VENV)"
if [ ! -x "$VENV/bin/python" ]; then  # an update reuses the installed packages
  free_gb=$(df -Pk "$(dirname "$VENV")" | awk 'NR==2 {print int($4 / 1048576)}')
  [ "$free_gb" -ge 12 ] || fail "only $free_gb GB free for $VENV: the install needs about 12 GB (set KUKKIRI_VENV to another disk)"
fi

say "Installing into $VENV (about 10 GB; takes a few minutes)"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
PIP=("$VENV/bin/python" -m pip install --disable-pip-version-check)
"${PIP[@]}" -q --upgrade pip
"${PIP[@]}" -r "$REQ/requirements-torch.txt"
"${PIP[@]}" -r "$REQ/requirements-base.txt"
"${PIP[@]}" --no-deps -r "$REQ/requirements-nodeps.txt"
if [ "$GUI" = 1 ]; then
  "${PIP[@]}" "PySide6-Essentials>=6.8" "nvidia-ml-py>=12" "psutil>=5"
  qt=$("$VENV/bin/python" -c 'import os, PySide6; print(os.path.dirname(PySide6.__file__) + "/Qt")')
  missing=$(ldd "$qt/plugins/platforms/libqxcb.so" "$qt/lib/libQt6Widgets.so.6" 2> /dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')
  if [ -n "$missing" ]; then
    echo "warning: the app needs system libraries this machine lacks: $missing"
    echo "         Debian/Ubuntu: sudo apt install libegl1 libgl1 libfontconfig1 libdbus-1-3 libxkbcommon-x11-0 libxcb-cursor0 \\"
    echo "                        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xkb1"
  fi
fi
"${PIP[@]}" --no-deps --force-reinstall .

say "Checking the install"
"$VENV/bin/kukkiri" check | "$VENV/bin/python" -c '
import json, sys
marks = {"ok": "ok  ", "info": "    ", "warn": "WARN", "fail": "FAIL"}
for line in sys.stdin:
    try:
        r = json.loads(line)
    except ValueError:
        continue
    item = r["item"].format(**r.get("item_args", {}))
    detail = r["detail"].format(**r.get("args", {}))
    print("  " + marks.get(r["status"], "    ") + "  " + item + ": " + detail)
' || true

say "Done"
echo "Run it with:  $VENV/bin/kukkiri restore input.mp4"
[ "$GUI" = 1 ] && echo "The app:      $VENV/bin/kukkiri-gui"
echo "(or 'source $VENV/bin/activate' first and use kukkiri / kukkiri-gui)"
