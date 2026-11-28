from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QLocale, QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPalette
from kukkiri.cli import LTX_MIN_VRAM_GB, VIDEO_EXTS, _seconds, collect_videos
from kukkiri.constants import DISCORD_INVITES, HF_REPO, PUBLIC_REPO
from kukkiri.i18n import LANGUAGES, set_language, system_language, tr
from kukkiri.mux import partial_path
from kukkiri.split import BUSY_EXIT, job_dir
from kukkiri.weights import MODELS as ALL_MODELS
from kukkiri.weights import default_dir, download_size, is_ltx, missing, version_of

from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

MAC = sys.platform == "darwin"
MODELS = [m for m in ALL_MODELS if not (MAC and is_ltx(version_of(m)))]

_PROGRESS_RE = re.compile(r"^(\w+):\s+([\d.]+)%(?:, ([\d.]+) min left)?(?:, (\d+) fps)?")
_LOG_NOISE_RE = re.compile(
    r"INVALID INDEX|Unable to load package|^nanobind:|^- (leaked|\.\.\. skipped)"
    r"|rf-detr - (Using a different number of positional encodings|Using patch size|Checkpoint has \d+ classes)"
    r"|tensorrt\.plugin module is experimental|Unable to import quantiz|torch\._opaque_base is deprecated"
    r"|register_opaque_type is deprecated|typ='reference' is deprecated|Not enough SMs to use max_autotune"
    r"|^watch it while it restores:"
)
_NOISE_RE = re.compile(r"WARN|Warning|^W\d{4} |\[TRT\] \[W\]|^PyNvVideoCodec:|Unable to import quantiz")


def split_progress(line: str) -> list[str]:
    m = _PROGRESS_RE.match(line)
    rest = line[m.end():].strip() if m else ""
    return [line[:m.end()], rest] if rest else [line]


COLS = ["File", "", "Section", "Status", "Progress", "Speed", "Time left"]
_TRIM, _SECTION, _STATUS, _PROGRESS, _SPEED, _ETA = range(1, 7)

THEMES = {
    "light": dict(bg="#f6eedd", card="#fbf6ea", text="#221d1a", muted="#6b6056", rule="#e2d5bc", accent="#d8394f", code="#ede2ca"),
    "dark": dict(bg="#17130f", card="#1d1813", text="#f1e9e2", muted="#a59a91", rule="#2e2721", accent="#f4606f", code="#231d18"),
}
_QSS = """
QWidget {{ background: {bg}; color: {text}; font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; }}
QLineEdit, QPlainTextEdit, QTableWidget {{ background: {card}; border: 1px solid {rule}; border-radius: 6px;
    selection-background-color: {accent}; selection-color: #ffffff; }}
QLineEdit {{ padding: 5px 8px; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {muted}; }}
QPlainTextEdit {{ padding: 4px; font-family: Consolas, "SF Mono", Menlo, monospace; font-size: 9pt; }}
QPushButton {{ background: {card}; border: 1px solid {rule}; border-radius: 8px; padding: 6px 16px; font-weight: 600; }}
QPushButton:hover {{ border-color: {muted}; }}
QPushButton:pressed {{ background: {code}; }}
QPushButton:disabled {{ color: {muted}; }}
QPushButton#primary {{ background: {accent}; border-color: {accent}; color: #ffffff; }}
QPushButton#primary:hover {{ border-color: {text}; }}
QPushButton#primary:disabled {{ background: {rule}; border-color: {rule}; color: {muted}; }}
QPushButton:flat {{ background: transparent; border: none; color: {muted}; }}
QPushButton:flat:checked {{ color: {accent}; }}
QTableWidget {{ gridline-color: {rule}; }}
QTableWidget::item {{ padding: 4px; }}
QTableWidget::item:selected {{ background: {code}; color: {text}; }}
QHeaderView::section {{ background: {code}; color: {muted}; border: none; border-bottom: 1px solid {rule};
    border-right: 1px solid {rule}; padding: 6px; font-weight: 600; }}
QTableCornerButton::section {{ background: {code}; border: none; }}
QProgressBar {{ background: {code}; border: none; border-radius: 5px; margin: 8px 8px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 5px; }}
QComboBox {{ background: {card}; border: 1px solid {rule}; border-radius: 6px; padding: 4px 8px; }}
QComboBox:focus {{ border-color: {muted}; }}
QComboBox QAbstractItemView {{ background: {card}; selection-background-color: {code}; selection-color: {text}; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {muted}; border-radius: 3px; background: {card}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
QToolButton {{ background: transparent; border: none; padding: 2px; }}
QToolButton:hover {{ color: {accent}; }}
QToolButton:disabled {{ color: {rule}; }}
QStatusBar {{ color: {muted}; }}
QStatusBar::item {{ border: none; }}
QMessageBox, QDialog {{ background: {bg}; }}
"""


def apply_theme(app: QApplication) -> None:
    dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    t = THEMES["dark" if dark else "light"]
    app.setStyle("Fusion")
    pal = app.palette()
    for role, key in ((QPalette.ColorRole.Window, "bg"), (QPalette.ColorRole.Base, "card"), (QPalette.ColorRole.Text, "text"),
                      (QPalette.ColorRole.WindowText, "text"), (QPalette.ColorRole.Button, "card"),
                      (QPalette.ColorRole.ButtonText, "text"), (QPalette.ColorRole.Highlight, "accent"),
                      (QPalette.ColorRole.PlaceholderText, "muted"), (QPalette.ColorRole.AlternateBase, "code")):
        pal.setColor(role, QColor(t[key]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(_QSS.format(**t))


@dataclass
class Job:
    src: Path
    start: str
    duration: str
    status: str = "Waiting"
    status_args: dict | None = None
    output: Path | None = None


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _section_text(job: Job) -> str:
    if job.duration:
        start = _seconds(job.start or "0")
        return f"{_clock(start)} - {_clock(start + _seconds(job.duration))}"
    return tr("from {start}", start=job.start) if job.start else tr("whole video")


class SectionDialog(QDialog):

    def __init__(self, parent, job: Job) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Restore only part of this video"))
        self.setMinimumWidth(380)
        self.start = QLineEdit(job.start)
        self.start.setPlaceholderText(tr("from the beginning (e.g. 37:45)"))
        self.duration = QLineEdit(job.duration)
        self.duration.setPlaceholderText(tr("to the end (e.g. 0:30 for 30 seconds)"))
        form = QFormLayout()
        form.addRow(tr("Start"), self.start)
        form.addRow(tr("Duration"), self.duration)
        self.error = QLabel("")
        buttons = QDialogButtonBox()
        buttons.addButton(tr("OK"), QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(tr("Cancel"), QDialogButtonBox.ButtonRole.RejectRole)
        whole = buttons.addButton(tr("Whole video"), QDialogButtonBox.ButtonRole.ResetRole)
        whole.clicked.connect(lambda: (self.start.clear(), self.duration.clear(), self.accept()))
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.error)
        lay.addWidget(buttons)

    def _ok(self) -> None:
        try:
            for t in (self.start.text().strip(), self.duration.text().strip()):
                if t:
                    _seconds(t)
        except ValueError:
            self.error.setText(tr("Times are seconds, mm:ss or hh:mm:ss."))
            return
        self.accept()


def worth_updating(tag: str, installed: str) -> bool:
    def parts(v):
        return tuple(int(x) for x in v.lstrip("v").split("+")[0].split("."))[:2]

    try:
        return parts(tag) > parts(installed)
    except ValueError:
        return False


def support_links() -> list[tuple[str, str]]:
    links = [("GitHub", f"https://github.com/{PUBLIC_REPO}/issues"),
             ("Hugging Face", f"https://huggingface.co/{HF_REPO}/discussions")]
    if len(DISCORD_INVITES) == 1:
        links.append((tr("Discord (unofficial)"), DISCORD_INVITES[0]))
    else:
        links += [(tr("Discord {n} (unofficial)", n=i), url) for i, url in enumerate(DISCORD_INVITES, 1)]
    return links


class SupportDialog(QDialog):

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Support"))
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)
        for name, url in support_links():
            link = QLabel(f'<a href="{url}">{name}</a>')
            link.setOpenExternalLinks(True)
            link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            layout.addWidget(link)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class CheckDialog(QDialog):

    MARKS = {"ok": ("✓", "#3a9a5b"), "info": ("•", None), "warn": ("!", "#d08a1c"), "fail": ("✗", "#d8394f")}

    def __init__(self, parent, args: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("System check"))
        self.setMinimumWidth(560)
        self.args = args
        self.grid = QGridLayout()
        self.grid.setColumnStretch(2, 1)
        buttons = QDialogButtonBox()
        self.again = buttons.addButton(tr("Check again"), QDialogButtonBox.ButtonRole.ActionRole)
        self.again.clicked.connect(self.run)
        buttons.addButton(tr("Close"), QDialogButtonBox.ButtonRole.RejectRole)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(self.grid)
        lay.addStretch(1)
        lay.addWidget(buttons)
        self.proc: QProcess | None = None
        self.run()

    def _clear(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()

    def run(self) -> None:
        self._clear()
        self.grid.addWidget(QLabel(tr("Checking…")), 0, 0, 1, 3)
        self.again.setEnabled(False)
        self.proc = QProcess(self)
        self.proc.finished.connect(self._done)
        self.proc.start(os.environ.get("KUKKIRI_WORKER_PYTHON", sys.executable), self.args)

    def _done(self, code: int, _status) -> None:
        out = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        err = bytes(self.proc.readAllStandardError()).decode(errors="replace")
        self._clear()
        rows = []
        for line in out.splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        if code != 0 or not rows:
            self.grid.addWidget(QLabel(tr("The check could not run (exit {code}):", code=code)), 0, 0, 1, 3)
            tail = QLabel((err or out).strip()[-600:])
            tail.setWordWrap(True)
            self.grid.addWidget(tail, 1, 0, 1, 3)
        for i, r in enumerate(rows):
            mark, color = self.MARKS.get(r["status"], ("•", None))
            m = QLabel(mark)
            if color:
                m.setStyleSheet(f"color: {color}; font-weight: 700;")
            name = QLabel(tr(r["item"], **r.get("item_args", {})))
            name.setStyleSheet("font-weight: 600;")
            detail = QLabel(tr(r["detail"], **r.get("args", {})))
            detail.setWordWrap(True)
            detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            for c, w in enumerate((m, name, detail)):
                self.grid.addWidget(w, i, c, Qt.AlignmentFlag.AlignTop)
        self.again.setEnabled(True)
        self.proc = None


class Meters:

    def __init__(self, gpu: int = 0) -> None:
        self.gpu = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        except Exception:
            pass
        try:
            import psutil

            self.psutil = psutil
            psutil.cpu_percent(None)
        except Exception:
            self.psutil = None

    def text(self) -> str:
        parts = []
        if self.gpu is not None:
            try:
                util = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu
                mem = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu)
                parts += [f"GPU {util}%", f"VRAM {mem.used / 1e9:.1f}/{mem.total / 1e9:.0f} GB"]
            except Exception:
                pass
        if self.psutil is not None:
            parts += [f"CPU {self.psutil.cpu_percent(None):.0f}%", f"RAM {self.psutil.virtual_memory().percent:.0f}%"]
        return "   ".join(parts)


def _output_name(job: Job, model: str) -> str:
    if not (job.start or job.duration):
        return f"{job.src.stem}.kukkiri-{model}.mp4"
    tag = "_".join(t.replace(":", "-") for t in (job.start or "0", job.duration) if t)
    return f"{job.src.stem}.kukkiri-{model}.{tag}.mp4"


APP_DEFAULT_MODEL = "kiyo-ltx-v1"


ALL_GPUS = -1


def _nvidia_gpus() -> list[tuple[int, str, float]]:
    try:
        import pynvml

        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            out.append((i, name.decode() if isinstance(name, bytes) else name, pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30))
        return out
    except Exception:
        return []


def _default_model(vr: bool, gpus: list[tuple[int, str, float]], gpu: int) -> str:
    gb = next((g for i, _, g in gpus if i == gpu), 0.0)
    return APP_DEFAULT_MODEL if not vr and APP_DEFAULT_MODEL in MODELS and gb >= LTX_MIN_VRAM_GB else MODELS[0]


STAGED_LTX_GB = 15


def find_player() -> list[str] | None:
    import shutil

    if sys.platform == "win32":
        found = []
        try:
            import winreg

            for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\vlc.exe") as k:
                        found.append(winreg.QueryValue(k, None))
                except OSError:
                    pass
        except ImportError:
            pass
        found += [str(Path(os.environ[v]) / "VideoLAN" / "VLC" / "vlc.exe")
                  for v in ("ProgramFiles", "ProgramFiles(x86)") if os.environ.get(v)]
        found.append(shutil.which("vlc"))
        player = next((f for f in found if f and Path(f).is_file()), None)
        return [player] if player else None
    if sys.platform == "darwin":
        return ["open", "-a", "VLC"] if Path("/Applications/VLC.app").exists() else None
    player = shutil.which("vlc")
    return [player] if player else None


def _remove_partial(output: Path | None) -> None:
    if output:
        try:
            partial_path(output).unlink(missing_ok=True)
        except OSError:
            pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        from kukkiri import __version__

        self.setWindowTitle(f"kukkiri {__version__}")
        if not QApplication.instance().property("kukkiri_theme"):
            apply_theme(QApplication.instance())
            QApplication.instance().setProperty("kukkiri_theme", True)
        self.resize(1040, 640)
        self.setAcceptDrops(True)
        self.settings = QSettings("kukkiri", "kukkiri")
        set_language(self.settings.value("language", "") or system_language(QLocale.system().name()))
        self.jobs: list[Job] = []
        self.proc: QProcess | None = None
        self.current: int | None = None
        self.cancelled = False
        self._partial = ""
        self._skip_reason = ""
        self._texts: list[tuple] = []
        self._hint = "Add videos or drop them here. Double-click a row to watch it while it is restored, or to open it when done."
        self._status_key = self._hint

        self.language = QComboBox()
        for code, name in LANGUAGES.items():
            self.language.addItem(name, code)
        self.language.setCurrentIndex(self.language.findData(self.settings.value("language", "")
                                                             or system_language(QLocale.system().name())))
        self.language.currentIndexChanged.connect(self._language_changed)
        self.check_btn = QPushButton()
        self.check_btn.setFlat(True)
        self.check_btn.clicked.connect(self._system_check)
        self.support_btn = QPushButton()
        self.support_btn.setFlat(True)
        self.support_btn.clicked.connect(lambda: SupportDialog(self).exec())
        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self.support_btn)
        top.addWidget(self.check_btn)
        top.addWidget(self.language)

        self.model = QComboBox()
        for m in MODELS:
            self.model.addItem(m, m)
        saved = self.settings.value("model", "")
        self.gpus = _nvidia_gpus()
        self.gpu = QComboBox()
        for i, name, gb in self.gpus:
            self.gpu.addItem(f"GPU {i}: {name} ({gb:.0f} GB)", i)
        if len(self.gpus) > 1:
            self.gpu.addItem(tr("All GPUs (each video split across them)"), ALL_GPUS)
            self._t(lambda text: self.gpu.setItemText(self.gpu.count() - 1, text), "All GPUs (each video split across them)")
        self.gpu.setCurrentIndex(max(0, self.gpu.findData(int(self.settings.value("gpu", 0) or 0))))
        self.gpu.currentIndexChanged.connect(self._gpu_changed)
        start = saved if saved in MODELS else _default_model(bool(self.settings.value("vr", "")), self.gpus, max(0, self._gpu()))
        self.model.setCurrentIndex(max(0, self.model.findData(start)))
        self.weights = QLineEdit(self.settings.value("weights", "") or str(default_dir(self._version())))
        self.model.currentIndexChanged.connect(self._model_changed)
        self.outdir = QLineEdit(self.settings.value("outdir", ""))
        self.sensitivity = QComboBox()
        for key, value in (("Normal", "normal"), ("High (also finds faint or small mosaics)", "high"),
                           ("Low (fewer false detections)", "low")):
            self.sensitivity.addItem(tr(key), value)
            self._t(lambda text, i=self.sensitivity.count() - 1: self.sensitivity.setItemText(i, text), key)
        self.sensitivity.setCurrentIndex(max(0, self.sensitivity.findData(self.settings.value("sensitivity", "normal"))))
        self.sensitivity.currentIndexChanged.connect(
            lambda: self.settings.setValue("sensitivity", self.sensitivity.currentData()))
        self.vr = QComboBox()
        for key, value in (("No (flat video)", ""), ("Side-by-side 180°", "sbs"),
                           ("Side-by-side 180°, fisheye mosaic (blocks bend into fans near the bottom)", "sbs-fisheye")):
            self.vr.addItem(tr(key), value)
            self._t(lambda text, i=self.vr.count() - 1: self.vr.setItemText(i, text), key)
        self.vr.setCurrentIndex(max(0, self.vr.findData(self.settings.value("vr", ""))))
        self.vr.currentIndexChanged.connect(lambda: self.settings.setValue("vr", self.vr.currentData()))
        self.codec = QComboBox()
        for key, value in (("H.264 (plays everywhere)", "h264"), ("HEVC (smaller files, needs a newer player)", "hevc")):
            self.codec.addItem(tr(key), value)
            self._t(lambda text, i=self.codec.count() - 1: self.codec.setItemText(i, text), key)
        self.codec.setCurrentIndex(max(0, self.codec.findData(self.settings.value("codec", "h264"))))
        self.codec.currentIndexChanged.connect(lambda: self.settings.setValue("codec", self.codec.currentData()))
        self.max_1080p = QCheckBox()
        self.skip_check = QCheckBox()
        self.keep_fps = QCheckBox()
        for box, key in ((self.max_1080p, "max_1080p"), (self.skip_check, "skip_check"), (self.keep_fps, "keep_fps")):
            box.setChecked(self.settings.value(key, "false") == "true")
            box.toggled.connect(lambda on, k=key: self.settings.setValue(k, "true" if on else "false"))
        self._t(self.max_1080p.setText, "Limit to 1080p (faster for 4K videos)")
        self._t(self.skip_check.setText, "Restore even if a video looks uncensored")
        self._t(self.keep_fps.setText, "Keep 60 fps videos at their full frame rate (kiyo-ltx, about twice as slow)")
        form = QFormLayout()
        for key, field in (("Model", self.model), ("Weights folder", self._with_browse(self.weights, "weights")),
                           ("Output folder", self._with_browse(self.outdir, "outdir")),
                           ("Detection sensitivity", self.sensitivity), *([] if MAC else [("VR video", self.vr)]),
                           ("Video format", self.codec), *([("GPU", self.gpu)] if len(self.gpus) > 1 else []),
                           ("", self.max_1080p),
                           ("", self.skip_check),
                           ("", self.keep_fps)):
            label = QLabel()
            if key:
                self._t(label.setText, key)
            form.addRow(label, field)
            form.setRowVisible(form.rowCount() - 1, False)

        self.table = QTableWidget(0, len(COLS))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(_TRIM, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(_TRIM, 34)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._open_output)

        self.add_btn = QPushButton()
        self.add_btn.clicked.connect(self._pick_files)
        self.folder_btn = QPushButton()
        self.folder_btn.clicked.connect(self._pick_folder)
        self.remove_btn = QPushButton()
        self.remove_btn.clicked.connect(self._remove_selected)
        self.advanced_btn = QPushButton()
        self.advanced_btn.setFlat(True)
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.toggled.connect(lambda on: [form.setRowVisible(row, on) for row in range(form.rowCount())])
        self.advanced_btn.setChecked(bool(self.outdir.text()))
        self.previews = QCheckBox()
        self.previews.setChecked(self.settings.value("previews", "false") == "true")
        self.previews.toggled.connect(lambda on: self.settings.setValue("previews", "true" if on else "false"))
        self.run_btn = QPushButton()
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._start_queue)
        self.cancel_btn = QPushButton()
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        self.watch_btn = QPushButton()
        self.watch_btn.clicked.connect(lambda: self._watch(self.current))
        self.watch_btn.setEnabled(False)
        buttons = QHBoxLayout()
        for b in (self.add_btn, self.folder_btn, self.remove_btn, self.advanced_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        for b in (self.previews, self.watch_btn, self.run_btn, self.cancel_btn):
            buttons.addWidget(b)
        self._watch_index = None
        self._watched: set = set()
        self._leftovers: list[Path] = []
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._update_watch)
        self._watch_timer.start(1000)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFixedHeight(130)
        self._full_log: list[str] = []
        self._log_stage, self._log_tick = None, 0.0
        log_label = QLabel()
        self.save_log_btn = QPushButton()
        self.save_log_btn.setFlat(True)
        self.save_log_btn.clicked.connect(self._save_log)
        log_row = QHBoxLayout()
        log_row.addWidget(log_label)
        log_row.addStretch(1)
        log_row.addWidget(self.save_log_btn)

        for setter, key in ((self.check_btn.setText, "System check"), (self.add_btn.setText, "Add videos"),
                            (self.folder_btn.setText, "Add folder"), (self.remove_btn.setText, "Remove"),
                            (self.advanced_btn.setText, "Advanced"),
                            (self.previews.setText, "Before/after previews"),
                            (self.run_btn.setText, "Start"), (self.cancel_btn.setText, "Stop"),
                            (self.watch_btn.setText, "Watch"),
                            (self.watch_btn.setToolTip, "Play the video being restored while it is restored, in VLC"),
                            (log_label.setText, "Log"), (self.outdir.setPlaceholderText, "next to each input"),
                            (self.save_log_btn.setText, "Save log"),
                            (self.support_btn.setText, "Support"),
                            (self.previews.setToolTip, "Also write up to 3 short side-by-side clips (with a kukkiri "
                             "watermark) of the most restored moments, next to each restored video")):
            self._t(setter, key)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(form)
        root.addWidget(self.table, 1)
        root.addLayout(buttons)
        root.addLayout(log_row)
        root.addWidget(self.log)
        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.meters = Meters(max(0, self._gpu()))
        self.meter_label = QLabel()
        self.statusBar().addPermanentWidget(self.meter_label)
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: self.meter_label.setText(self.meters.text()))
        self._timer.start(2000)
        self._retranslate()
        QTimer.singleShot(2000, self._check_update)


    def _check_update(self) -> None:
        from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest

        from kukkiri import __version__

        self._installed = __version__
        self._net = QNetworkAccessManager(self)
        req = QNetworkRequest(QUrl(f"https://api.github.com/repos/{PUBLIC_REPO}/releases/latest"))
        req.setRawHeader(b"Accept", b"application/vnd.github+json")
        req.setTransferTimeout(5000)
        reply = self._net.get(req)
        reply.finished.connect(lambda: self._update_reply(reply))

    def _update_reply(self, reply) -> None:
        reply.deleteLater()
        try:
            release = json.loads(bytes(reply.readAll()).decode())
            latest, url = release["tag_name"], release["html_url"]
        except (ValueError, KeyError, TypeError):
            return
        if worth_updating(latest, self._installed):
            box = QMessageBox(self)
            box.setWindowTitle("kukkiri")
            box.setText(tr("kukkiri {latest} is available (this is {installed}).", latest=latest.lstrip("v"),
                           installed=self._installed))
            open_btn = box.addButton(tr("Open release page"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("Later"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is open_btn:
                QDesktopServices.openUrl(QUrl(url))


    def _t(self, setter, key: str) -> None:
        self._texts.append((setter, key))
        setter(tr(key))

    def _language_changed(self) -> None:
        code = self.language.currentData()
        self.settings.setValue("language", code)
        set_language(code)
        self._retranslate()

    def _retranslate(self) -> None:
        for setter, key in self._texts:
            setter(tr(key))
        self.table.setHorizontalHeaderLabels([tr(c) if c else "" for c in COLS])
        for i, m in enumerate(MODELS):
            if is_ltx(version_of(m)):
                self.model.setItemText(i, f"{m} ({tr('best quality, needs an 8 GB GPU')})")
        for row, job in enumerate(self.jobs):
            self.table.item(row, _SECTION).setText(_section_text(job))
            self.table.item(row, _STATUS).setText(tr(job.status, **(job.status_args or {})))
            self.table.cellWidget(row, _TRIM).setToolTip(tr("Restore only part of this video"))
        self.statusBar().showMessage(tr(self._status_key))

    def _status(self, key: str) -> None:
        self._status_key = key
        self.statusBar().showMessage(tr(key))

    def _log_full(self, line: str) -> None:
        if _LOG_NOISE_RE.search(line):
            return
        from datetime import datetime

        self._full_log.append(f"{datetime.now():%H:%M:%S} {line}")
        del self._full_log[:-20000]

    def _save_log(self) -> None:
        import platform
        from datetime import datetime

        from kukkiri import __version__

        default = str(Path.home() / f"kukkiri-log-{datetime.now():%Y%m%d-%H%M%S}.txt")
        path, _ = QFileDialog.getSaveFileName(self, tr("Save log"), default, "Text (*.txt)")
        if not path:
            return
        head = [f"kukkiri {__version__}, model {self.model.currentData()}, {platform.platform()}, "
                f"Python {platform.python_version()}", ""]
        Path(path).write_text("\n".join(head + self._full_log) + "\n", encoding="utf-8")
        self.statusBar().showMessage(tr("Log saved to {path}", path=path), 8000)

    def _system_check(self) -> None:
        CheckDialog(self, ["-m", "kukkiri", "check", "--weights", self.weights.text(),
                           "--model", self.model.currentData(),
                           *(["--gpu", "0"] if self._gpu() == ALL_GPUS else self._gpu_args())]).exec()

    def _gpu(self) -> int:
        return self.gpu.currentData() if self.gpu.count() else 0

    def _gpu_args(self) -> list[str]:
        if len(self.gpus) < 2:
            return []
        return ["--gpus", "all"] if self._gpu() == ALL_GPUS else ["--gpu", str(self._gpu())]

    def _gpu_changed(self) -> None:
        self.settings.setValue("gpu", self._gpu())
        if hasattr(self, "meters"):
            self.meters = Meters(max(0, self._gpu()))


    def _version(self) -> str:
        return version_of(self.model.currentData())

    def _model_changed(self) -> None:
        self.settings.setValue("model", self.model.currentData())
        if not self.settings.value("weights", ""):
            self.weights.setText(str(default_dir(self._version())))

    def _with_browse(self, edit: QLineEdit, key: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(edit)
        b = QPushButton()
        self._t(b.setText, "Browse")

        def browse():
            d = QFileDialog.getExistingDirectory(self, tr("Choose folder"), edit.text() or str(Path.home()))
            if d:
                edit.setText(d)
                self.settings.setValue(key, d)

        b.clicked.connect(browse)
        if key != "weights":
            edit.textChanged.connect(lambda text: self.settings.setValue(key, text))
        row.addWidget(b)
        return row


    def add_files(self, paths: list[str]) -> None:
        for path in collect_videos(paths):
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            job = Job(path, "", "")
            self.jobs.append(job)
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, text in enumerate([path.name, "", _section_text(job), tr(job.status), "", "", ""]):
                self.table.setItem(r, c, QTableWidgetItem(text))
            trim = QToolButton()
            trim.setText("✂")
            trim.setToolTip(tr("Restore only part of this video"))
            trim.clicked.connect(lambda _=False, j=job: self._edit_section(j))
            self.table.setCellWidget(r, _TRIM, trim)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            self.table.setCellWidget(r, _PROGRESS, bar)

    def set_section(self, row: int, start: str, duration: str) -> None:
        job = self.jobs[row]
        job.start, job.duration = start.strip(), duration.strip()
        self.table.item(row, _SECTION).setText(_section_text(job))

    def _edit_section(self, job: Job) -> None:
        if job.status != "Waiting":
            return
        d = SectionDialog(self, job)
        if d.exec() == QDialog.DialogCode.Accepted and job in self.jobs:
            self.set_section(self.jobs.index(job), d.start.text(), d.duration.text())

    def _last_folder(self) -> str:
        last = self.settings.value("last_folder", "") or ""
        return last if last and Path(last).is_dir() else str(Path.home())

    def _pick_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, tr("Add videos"), self._last_folder(), f"{tr('Videos')} ({exts})")
        if files:
            self.settings.setValue("last_folder", str(Path(files[0]).parent))
        self.add_files(files)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, tr("Add a folder of videos"), self._last_folder())
        if d:
            self.settings.setValue("last_folder", str(Path(d).parent))
            self.add_files([d])

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_files([u.toLocalFile() for u in e.mimeData().urls()])

    def _remove_selected(self) -> None:
        import shutil

        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            if r == self.current:
                continue
            job = self.jobs[r]
            if job.status == "Stopped" and job.output:
                from kukkiri.split import JobBusy, _lock, _unlock

                try:
                    held = _lock(job_dir(job.output))
                except (JobBusy, OSError):
                    held = None
                if held is not None:
                    try:
                        shutil.rmtree(job_dir(job.output), ignore_errors=True)
                        _remove_partial(job.output)
                    finally:
                        _unlock(job_dir(job.output), held)
            self.table.removeRow(r)
            del self.jobs[r]
            if self.current is not None and r < self.current:
                self.current -= 1

    def _set(self, row: int, status: str | None = None, pct: float | None = None, eta: str | None = None,
             speed: str | None = None, **status_args) -> None:
        if status is not None:
            job = self.jobs[row]
            job.status, job.status_args = status, status_args or None
            self.table.item(row, _STATUS).setText(tr(status, **status_args))
            self.table.cellWidget(row, _TRIM).setEnabled(status == "Waiting")
        if pct is not None:
            self.table.cellWidget(row, _PROGRESS).setValue(int(pct * 10))
        if eta is not None:
            self.table.item(row, _ETA).setText(eta)
        if speed is not None:
            self.table.item(row, _SPEED).setText(speed)


    def _start_queue(self) -> None:
        weights = Path(self.weights.text())
        if missing(weights, self._version()):
            msg = tr("kukkiri needs the {model} model weights (about {size}). Download them now into\n{weights}?",
                     model=self.model.currentData(), weights=weights, size=download_size(self._version()))
            if QMessageBox.question(self, "kukkiri", msg) != QMessageBox.StandardButton.Yes:
                return
        for i, j in enumerate(self.jobs):
            if j.status not in ("Waiting", "Stopped", "Done"):
                self._set(i, "Waiting", 0, "", "")
        self.run_btn.setEnabled(False)
        self.cancelled = False
        self._status(self._hint)
        self._next()

    def _next(self) -> None:
        waiting = [i for i, j in enumerate(self.jobs) if j.status in ("Waiting", "Stopped")]
        self.current = waiting[0] if waiting and not self.cancelled else None
        if self.current is None:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self._status("Stopped." if self.cancelled else "Queue finished.")
            return
        job = self.jobs[self.current]
        outdir = Path(self.outdir.text()) if self.outdir.text() else job.src.parent
        model = self.model.currentData()
        job.output = outdir / _output_name(job, model)
        n = 2
        while job.output.exists():
            job.output = outdir / _output_name(job, model).replace(".mp4", f" ({n}).mp4")
            n += 1
        args = ["-m", "kukkiri", "restore", str(job.src), "-o", str(job.output), "--weights", self.weights.text(),
                "--model", model]
        if self.previews.isChecked():
            args.append("--previews")
        if self.sensitivity.currentData() != "normal":
            args += ["--sensitivity", self.sensitivity.currentData()]
        if self.max_1080p.isChecked():
            args.append("--max-1080p")
        if self.skip_check.isChecked():
            args.append("--skip-check")
        if self.keep_fps.isChecked():
            args.append("--keep-fps")
        if self.vr.currentData() and not MAC:
            args += ["--vr", "sbs"] + (["--vr-lens", "fisheye"] if self.vr.currentData() == "sbs-fisheye" else [])
        args += self._gpu_args()
        if self.codec.currentData() != "h264":
            args += ["--codec", self.codec.currentData()]
        if job.start:
            args += ["--start", job.start]
        if job.duration:
            args += ["--duration", job.duration]
        self._partial = ""
        self._skip_reason = ""
        self._done = False
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONWARNINGS", "ignore")
        env.insert("KUKKIRI_KEEP_PART", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self._log_stage = None
        self._log_full(f"--- {job.src.name} -> {job.output}")
        self._log_full("$ kukkiri " + " ".join(args[2:] if args[:2] == ["-m", "kukkiri"] else args))
        self.log.appendPlainText(f"--- {job.src.name} -> {job.output}")
        self._set(self.current, "Starting", 0, "", "")
        self.cancel_btn.setEnabled(True)
        self.proc.start(os.environ.get("KUKKIRI_WORKER_PYTHON", sys.executable), args)

    def _read(self) -> None:
        text = self._partial + bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        parts = re.split(r"[\r\n]", text)
        self._partial = parts.pop()
        for line in (q for p in parts for q in split_progress(p.strip())):
            if not line:
                continue
            m = _PROGRESS_RE.match(line)
            if not m:
                self._log_full(line)
            if m:
                stage, pct, eta, fps = m.group(1), float(m.group(2)), m.group(3), m.group(4)
                now = time.monotonic()
                if stage != self._log_stage or now - self._log_tick >= 30:
                    self._log_stage, self._log_tick = stage, now
                    self._log_full(line)
                label = {"remaster": "Restoring", "remux": "Finishing", "preparing": "Preparing",
                         "download": "Downloading weights", "detect": "Finding mosaics",
                         "ltx": "Restoring with LTX"}.get(stage, stage)
                self._set(self.current, label, pct, tr("{n} min", n=f"{float(eta):.1f}") if eta else "",
                          f"{fps} fps" if fps else "")
                continue
            if line.startswith("done:"):
                self._done = True
            elif line.startswith("skipped:"):
                self._skip_reason = line[len("skipped:"):].strip()
            elif line.startswith("check:"):
                self._set(self.current, "Checking")
            elif line.startswith("cutting"):
                self._set(self.current, "Cutting section")
            if not _NOISE_RE.search(line) and not _LOG_NOISE_RE.search(line):
                self.log.appendPlainText(line)

    def _finished(self, code: int, _status) -> None:
        row, job = self.current, self.jobs[self.current]
        self._end_watch(row, done=(code == 0 or self._done) and not (self.cancelled and not self._done))
        if self.cancelled and not self._done:
            self._set(row, "Stopped", 0, "", "")
            if job.output and job.output.exists():
                job.output.unlink()
            _remove_partial(job.output)
        elif code == 0 or self._done:
            self._set(row, "Done", 100, "", "")
        elif code == 2:
            reason = self._skip_reason.split("(")[0].strip() or "nothing to restore"
            self._set(row, "Skipped: {reason}", 0, "", "", reason=tr(reason))
        else:
            self._set(row, "Failed (exit {code}), see log", None, "", "", code=code)
            if code != BUSY_EXIT:
                _remove_partial(job.output)
        self.proc = None
        self._next()

    def _cancel(self) -> None:
        if self.proc is None:
            return
        self.cancelled = True
        self.proc.terminate()
        if not self.proc.waitForFinished(5000):
            self.proc.kill()

    def _open_output(self, row: int, col: int) -> None:
        if col == _TRIM:
            return
        if row == self.current and self.proc is not None:
            self._watch(row)
            return
        out = self.jobs[row].output
        if out and out.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _watchable(self, row: int | None) -> Path | None:
        from kukkiri.watch import FragmentIndex, playlist_path

        if row is None or row != self.current or self.proc is None or not self.jobs[row].output:
            return None
        part = partial_path(self.jobs[row].output)
        if self._watch_index is None or self._watch_index.part != part:
            self._watch_index = FragmentIndex(part)
        if not part.exists():
            return None
        playlist = playlist_path(part)
        try:
            return playlist if self._watch_index.write(playlist) else None
        except OSError:
            return None

    def _update_watch(self) -> None:
        self.watch_btn.setEnabled(self._watchable(self.current) is not None)
        tip = tr("Play the video being restored while it is restored, in VLC")
        gpu = self._gpu()
        gbs = [gb for i, _, gb in self.gpus if gpu in (i, ALL_GPUS)]
        if is_ltx(version_of(self.model.currentData())) and gbs and min(gbs) < STAGED_LTX_GB:
            tip += "\n" + tr("With kiyo-ltx on a GPU with less than 15 GB of memory the video is restored in passes and "
                             "written in the last one: Watch turns on near the end.")
        if self.watch_btn.toolTip() != tip:
            self.watch_btn.setToolTip(tip)

    def _end_watch(self, row: int, done: bool) -> None:
        from kukkiri.watch import playlist_path

        job, index, self._watch_index = self.jobs[row], self._watch_index, None
        self._clean_leftovers()
        if not job.output:
            return
        part = partial_path(job.output)
        playlist = playlist_path(part)
        files = [part, playlist, playlist.with_name(playlist.name + ".tmp")]
        if done and id(job) in self._watched and index is not None and index.part == part and part.exists():
            index.write(playlist, finished=True)
            self._leftovers += files
            return
        self._leftovers += files
        self._clean_leftovers()

    def _clean_leftovers(self) -> None:
        keep = []
        for p in self._leftovers:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                keep.append(p)
        self._leftovers = keep

    def _watch(self, row: int | None) -> None:
        part = self._watchable(row)
        if part is None:
            return
        self._watched.add(id(self.jobs[row]))
        player = find_player()
        if player is None:
            QMessageBox.information(self, "kukkiri", tr(
                "To watch a video while it is restored, install VLC (videolan.org): kukkiri opens it there and it keeps "
                "playing as the restore goes on. The finished video plays in any player."))
            return
        QProcess.startDetached(player[0], player[1:] + [str(part)])

    def closeEvent(self, e):
        if self.proc is not None:
            if QMessageBox.question(self, "kukkiri", tr("A video is still being restored. Stop it and quit?")) != (
                QMessageBox.StandardButton.Yes
            ):
                e.ignore()
                return
            self._cancel()
        self._clean_leftovers()
        e.accept()


def main() -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("kukkiri.app")
    if MAC:
        os.environ["PATH"] = os.pathsep.join([os.environ.get("PATH", ""), "/opt/homebrew/bin", "/usr/local/bin"])
    app = QApplication(sys.argv)
    app.setApplicationName("kukkiri")
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("icon.png"))))
    win = MainWindow()
    win.add_files(sys.argv[1:])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
