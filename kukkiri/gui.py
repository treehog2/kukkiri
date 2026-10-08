from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QLocale, QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPalette
from kukkiri.cli import VIDEO_EXTS, _seconds, collect_videos
from kukkiri.i18n import LANGUAGES, set_language, system_language, tr
from kukkiri.weights import MODELS, default_dir, missing, version_of

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

_PROGRESS_RE = re.compile(r"^(\w+):\s+([\d.]+)%(?:, ([\d.]+) min left)?(?:, (\d+) fps)?")
_NOISE_RE = re.compile(r"WARN|Warning|^W\d{4} |\[TRT\] \[W\]|^PyNvVideoCodec:|Unable to import quantiz")
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


def _section_text(job: Job) -> str:
    if job.duration:
        return f"{job.start or '0'} + {job.duration}"
    return tr("from {start}", start=job.start) if job.start else tr("whole video")


class SectionDialog(QDialog):

    def __init__(self, parent, job: Job) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Restore only part of this video"))
        self.setMinimumWidth(380)
        self.start = QLineEdit(job.start)
        self.start.setPlaceholderText(tr("from the beginning (e.g. 37:45)"))
        self.duration = QLineEdit(job.duration)
        self.duration.setPlaceholderText(tr("to the end (e.g. 30)"))
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

    def __init__(self) -> None:
        self.gpu = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        if not QApplication.instance().property("kukkiri_theme"):
            apply_theme(QApplication.instance())
            QApplication.instance().setProperty("kukkiri_theme", True)
        self.setWindowTitle("kukkiri")
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
        self._status_key = "Add videos or drop them here. Double-click a finished row to open it."

        self.language = QComboBox()
        for code, name in LANGUAGES.items():
            self.language.addItem(name, code)
        self.language.setCurrentIndex(self.language.findData(self.settings.value("language", "")
                                                             or system_language(QLocale.system().name())))
        self.language.currentIndexChanged.connect(self._language_changed)
        self.check_btn = QPushButton()
        self.check_btn.setFlat(True)
        self.check_btn.clicked.connect(self._system_check)
        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self.check_btn)
        top.addWidget(self.language)

        self.model = QComboBox()
        for m in MODELS:
            self.model.addItem(m, m)
        saved = self.settings.value("model", "")
        self.model.setCurrentIndex(max(0, self.model.findData(saved)) if saved in MODELS else 0)
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
        self.max_1080p = QCheckBox()
        self.skip_check = QCheckBox()
        for box, key in ((self.max_1080p, "max_1080p"), (self.skip_check, "skip_check")):
            box.setChecked(self.settings.value(key, "false") == "true")
            box.toggled.connect(lambda on, k=key: self.settings.setValue(k, "true" if on else "false"))
        self._t(self.max_1080p.setText, "Limit to 1080p (faster for 4K videos)")
        self._t(self.skip_check.setText, "Restore even if a video looks uncensored")
        form = QFormLayout()
        for key, field in (("Model", self.model), ("Weights folder", self._with_browse(self.weights, "weights")),
                           ("Output folder", self._with_browse(self.outdir, "outdir")),
                           ("Detection sensitivity", self.sensitivity), ("", self.max_1080p), ("", self.skip_check)):
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
        buttons = QHBoxLayout()
        for b in (self.add_btn, self.folder_btn, self.remove_btn, self.advanced_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        for b in (self.previews, self.run_btn, self.cancel_btn):
            buttons.addWidget(b)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFixedHeight(130)
        log_label = QLabel()

        for setter, key in ((self.check_btn.setText, "System check"), (self.add_btn.setText, "Add videos…"),
                            (self.folder_btn.setText, "Add folder…"), (self.remove_btn.setText, "Remove"),
                            (self.advanced_btn.setText, "Advanced…"),
                            (self.previews.setText, "Before/after previews"),
                            (self.run_btn.setText, "Start"), (self.cancel_btn.setText, "Stop"),
                            (log_label.setText, "Log"), (self.outdir.setPlaceholderText, "next to each input"),
                            (self.previews.setToolTip, "Also write up to 3 short side-by-side clips (with a kukkiri "
                             "watermark) of the most restored moments, next to each restored video")):
            self._t(setter, key)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(form)
        root.addWidget(self.table, 1)
        root.addLayout(buttons)
        root.addWidget(log_label)
        root.addWidget(self.log)
        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.meters = Meters()
        self.meter_label = QLabel()
        self.statusBar().addPermanentWidget(self.meter_label)
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: self.meter_label.setText(self.meters.text()))
        self._timer.start(2000)
        self._retranslate()


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
        self.model.setItemText(0, f"{MODELS[0]} ({tr('latest')})")
        for row, job in enumerate(self.jobs):
            self.table.item(row, _SECTION).setText(_section_text(job))
            self.table.item(row, _STATUS).setText(tr(job.status, **(job.status_args or {})))
            self.table.cellWidget(row, _TRIM).setToolTip(tr("Restore only part of this video"))
        self.statusBar().showMessage(tr(self._status_key))

    def _status(self, key: str) -> None:
        self._status_key = key
        self.statusBar().showMessage(tr(key))

    def _system_check(self) -> None:
        CheckDialog(self, ["-m", "kukkiri", "check", "--weights", self.weights.text(),
                           "--model", self.model.currentData()]).exec()


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
        self._t(b.setText, "Browse…")

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

    def _pick_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, tr("Add videos"), str(Path.home()), f"{tr('Videos')} ({exts})")
        self.add_files(files)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, tr("Add a folder of videos"), str(Path.home()))
        if d:
            self.add_files([d])

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_files([u.toLocalFile() for u in e.mimeData().urls()])

    def _remove_selected(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            if r == self.current:
                continue
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
            msg = tr("kukkiri needs the {model} model weights (about 240 MB). Download them now into\n{weights}?",
                     model=self.model.currentData(), weights=weights)
            if QMessageBox.question(self, "kukkiri", msg) != QMessageBox.StandardButton.Yes:
                return
        self.run_btn.setEnabled(False)
        self.cancelled = False
        self._next()

    def _next(self) -> None:
        waiting = [i for i, j in enumerate(self.jobs) if j.status == "Waiting"]
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
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.log.appendPlainText(f"--- {job.src.name} -> {job.output}")
        self._set(self.current, "Starting", 0, "", "")
        self.cancel_btn.setEnabled(True)
        self.proc.start(os.environ.get("KUKKIRI_WORKER_PYTHON", sys.executable), args)

    def _read(self) -> None:
        text = self._partial + bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        parts = re.split(r"[\r\n]", text)
        self._partial = parts.pop()
        for line in (p.strip() for p in parts):
            if not line:
                continue
            m = _PROGRESS_RE.match(line)
            if m:
                stage, pct, eta, fps = m.group(1), float(m.group(2)), m.group(3), m.group(4)
                label = {"remaster": "Restoring", "remux": "Finishing", "preparing": "Preparing",
                         "download": "Downloading weights"}.get(stage, stage)
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
            if not _NOISE_RE.search(line):
                self.log.appendPlainText(line)

    def _finished(self, code: int, _status) -> None:
        row, job = self.current, self.jobs[self.current]
        if self.cancelled and not self._done:
            self._set(row, "Cancelled", 0, "", "")
            if job.output and job.output.exists():
                job.output.unlink()
        elif code == 0 or self._done:
            self._set(row, "Done", 100, "", "")
        elif code == 2:
            reason = self._skip_reason.split("(")[0].strip() or "nothing to restore"
            self._set(row, "Skipped: {reason}", 0, "", "", reason=tr(reason))
        else:
            self._set(row, "Failed (exit {code}), see log", None, "", "", code=code)
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
        out = self.jobs[row].output
        if out and out.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def closeEvent(self, e):
        if self.proc is not None:
            if QMessageBox.question(self, "kukkiri", tr("A video is still being restored. Stop it and quit?")) != (
                QMessageBox.StandardButton.Yes
            ):
                e.ignore()
                return
            self._cancel()
        e.accept()


def main() -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("kukkiri.app")
    app = QApplication(sys.argv)
    app.setApplicationName("kukkiri")
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("icon.png"))))
    win = MainWindow()
    win.add_files(sys.argv[1:])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
