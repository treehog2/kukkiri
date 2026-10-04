from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPalette
from kukkiri.cli import VIDEO_EXTS, collect_videos
from kukkiri.weights import default_dir, missing

from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFormLayout,
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
    QVBoxLayout,
    QWidget,
)

_PROGRESS_RE = re.compile(r"^(\w+):\s+([\d.]+)%(?:, ([\d.]+) min left)?")
_NOISE_RE = re.compile(r"WARN|Warning|^W\d{4} |\[TRT\] \[W\]|^PyNvVideoCodec:|Unable to import quantiz")
COLS = ["File", "Section", "Status", "Progress", "Time left"]

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
    output: Path | None = None


def _output_name(job: Job) -> str:
    if not (job.start or job.duration):
        return f"{job.src.stem}.kukkiri.mp4"
    tag = "_".join(t.replace(":", "-") for t in (job.start or "0", job.duration) if t)
    return f"{job.src.stem}.kukkiri.{tag}.mp4"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        if not QApplication.instance().property("kukkiri_theme"):
            apply_theme(QApplication.instance())
            QApplication.instance().setProperty("kukkiri_theme", True)
        self.setWindowTitle("kukkiri")
        self.resize(980, 620)
        self.setAcceptDrops(True)
        self.settings = QSettings("kukkiri", "kukkiri")
        self.jobs: list[Job] = []
        self.proc: QProcess | None = None
        self.current: int | None = None
        self.cancelled = False
        self._partial = ""
        self._skip_reason = ""

        self.weights = QLineEdit(self.settings.value("weights", "") or str(default_dir()))
        self.outdir = QLineEdit(self.settings.value("outdir", ""))
        self.outdir.setPlaceholderText("next to each input")
        self.start = QLineEdit()
        self.start.setPlaceholderText("from the beginning (e.g. 37:45)")
        self.duration = QLineEdit()
        self.duration.setPlaceholderText("to the end (e.g. 30)")
        form = QFormLayout()
        form.addRow("Weights folder", self._with_browse(self.weights, "weights"))
        form.setRowVisible(0, False)
        form.addRow("Output folder", self._with_browse(self.outdir, "outdir"))
        form.setRowVisible(1, False)
        section = QHBoxLayout()
        section.addWidget(QLabel("Start"))
        section.addWidget(self.start)
        section.addWidget(QLabel("Duration"))
        section.addWidget(self.duration)
        form.addRow("Section for new files", section)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._open_output)

        self.add_btn = QPushButton("Add videos…")
        self.add_btn.clicked.connect(self._pick_files)
        self.folder_btn = QPushButton("Add folder…")
        self.folder_btn.clicked.connect(self._pick_folder)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.advanced_btn = QPushButton("Advanced…")
        self.advanced_btn.setFlat(True)
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.toggled.connect(lambda on: (form.setRowVisible(0, on), form.setRowVisible(1, on)))
        self.advanced_btn.setChecked(bool(self.outdir.text()))
        self.run_btn = QPushButton("Start")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._start_queue)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        buttons = QHBoxLayout()
        for b in (self.add_btn, self.folder_btn, self.remove_btn, self.advanced_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        for b in (self.run_btn, self.cancel_btn):
            buttons.addWidget(b)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFixedHeight(130)

        root = QVBoxLayout()
        root.addLayout(form)
        root.addWidget(self.table, 1)
        root.addLayout(buttons)
        root.addWidget(QLabel("Log"))
        root.addWidget(self.log)
        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)
        self.statusBar().showMessage("Add videos or drop them here. Double-click a finished row to open it.")


    def _with_browse(self, edit: QLineEdit, key: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(edit)
        b = QPushButton("Browse…")

        def browse():
            d = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text() or str(Path.home()))
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
            job = Job(path, self.start.text().strip(), self.duration.text().strip())
            self.jobs.append(job)
            r = self.table.rowCount()
            self.table.insertRow(r)
            if job.duration:
                section = f"{job.start or '0'} + {job.duration}"
            else:
                section = f"from {job.start}" if job.start else "whole video"
            for c, text in enumerate([path.name, section, job.status, "", ""]):
                self.table.setItem(r, c, QTableWidgetItem(text))
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            self.table.setCellWidget(r, 3, bar)

    def _pick_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "Add videos", str(Path.home()), f"Videos ({exts})")
        self.add_files(files)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add a folder of videos", str(Path.home()))
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

    def _set(self, row: int, status: str | None = None, pct: float | None = None, eta: str | None = None) -> None:
        if status is not None:
            self.jobs[row].status = status
            self.table.item(row, 2).setText(status)
        if pct is not None:
            self.table.cellWidget(row, 3).setValue(int(pct * 10))
        if eta is not None:
            self.table.item(row, 4).setText(eta)


    def _start_queue(self) -> None:
        weights = Path(self.weights.text())
        if missing(weights):
            msg = f"kukkiri needs its model weights (about 240 MB). Download them now into\n{weights}?"
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
            self.statusBar().showMessage("Stopped." if self.cancelled else "Queue finished.")
            return
        job = self.jobs[self.current]
        outdir = Path(self.outdir.text()) if self.outdir.text() else job.src.parent
        job.output = outdir / _output_name(job)
        n = 2
        while job.output.exists():
            job.output = outdir / _output_name(job).replace(".mp4", f" ({n}).mp4")
            n += 1
        args = ["-m", "kukkiri", "restore", str(job.src), "-o", str(job.output), "--weights", self.weights.text()]
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
        self._set(self.current, "Starting", 0, "")
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
                stage, pct, eta = m.group(1), float(m.group(2)), m.group(3)
                label = {"remaster": "Restoring", "remux": "Finishing", "preparing": "Preparing",
                         "download": "Downloading weights"}.get(stage, stage)
                self._set(self.current, label, pct, f"{float(eta):.1f} min" if eta else "")
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
            self._set(row, "Cancelled", 0, "")
            if job.output and job.output.exists():
                job.output.unlink()
        elif code == 0 or self._done:
            self._set(row, "Done", 100, "")
        elif code == 2:
            self._set(row, f"Skipped: {self._skip_reason.split('(')[0].strip() or 'nothing to restore'}", 0, "")
        else:
            self._set(row, f"Failed (exit {code}), see log", None, "")
        self.proc = None
        self._next()

    def _cancel(self) -> None:
        if self.proc is None:
            return
        self.cancelled = True
        self.proc.terminate()
        if not self.proc.waitForFinished(5000):
            self.proc.kill()

    def _open_output(self, row: int, _col: int) -> None:
        out = self.jobs[row].output
        if out and out.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def closeEvent(self, e):
        if self.proc is not None:
            if QMessageBox.question(self, "kukkiri", "A video is still being restored. Stop it and quit?") != (
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
