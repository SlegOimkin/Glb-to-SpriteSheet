import json
import os
import re
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QAction, QPixmap, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QProgressBar, QCheckBox,
    QSplitter, QScrollArea, QSizePolicy
)


def find_blender_executable() -> str:
    env = os.environ.get("BLENDER_PATH")
    if env and Path(env).exists():
        return env
    p = shutil.which("blender")
    if p:
        return p
    if sys.platform == "darwin":
        mac = "/Applications/Blender.app/Contents/MacOS/Blender"
        if Path(mac).exists():
            return mac
    if sys.platform.startswith("win"):
        roots = [
            Path("C:/Program Files/Blender Foundation"),
            Path("C:/Program Files"),
        ]
        for root in roots:
            if root.exists():
                for exe in root.rglob("blender.exe"):
                    return str(exe)
    return ""


class PreviewLabel(QLabel):
    """Непрозрачный QLabel без артефактов + корректный рескейл превью при resize."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orig: QPixmap | None = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 240)
        self._user_canceled = False


        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(17, 17, 17))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(187, 187, 187))
        self.setPalette(pal)

        self.setText("No preview yet")

    def set_preview_pixmap(self, pix: QPixmap):
        self._orig = pix
        self._update_scaled()

    def _update_scaled(self):
        if self._orig is None or self._orig.isNull():
            super().setPixmap(QPixmap())
            return
        if self.width() <= 2 or self.height() <= 2:
            return
        scaled = self._orig.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._orig is not None:
            self._update_scaled()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GLB to SpriteSheet")

        self.proc = QProcess(self)
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.finished.connect(self._on_finished)

        self.stage = "idle"  # idle | probing | previewing | rendering | packing
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._preview_path = None

        self.anim_meta: dict[str, dict | None] = {}

        base = QWidget()
        self.setCentralWidget(base)
        root = QVBoxLayout(base)

        # ------- Project paths
        g_paths = QGroupBox("Project")
        fl = QFormLayout(g_paths)

        self.blenderPath = QLineEdit(find_blender_executable())
        btn_blender = QPushButton("Browse…")
        btn_blender.clicked.connect(self._browse_blender)
        hb_blender = QHBoxLayout()
        hb_blender.addWidget(self.blenderPath, 1)
        hb_blender.addWidget(btn_blender)
        fl.addRow("Blender:", hb_blender)

        self.glbPath = QLineEdit("")
        btn_glb = QPushButton("Browse…")
        btn_glb.clicked.connect(self._browse_glb)
        hb_glb = QHBoxLayout()
        hb_glb.addWidget(self.glbPath, 1)
        hb_glb.addWidget(btn_glb)
        fl.addRow(".glb file:", hb_glb)

        self.outDir = QLineEdit(str(Path.cwd() / "output"))
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_outdir)
        hb_out = QHBoxLayout()
        hb_out.addWidget(self.outDir, 1)
        hb_out.addWidget(btn_out)
        fl.addRow("Output folder:", hb_out)

        self.splitMode = QComboBox()
        self.splitMode.addItem("As is (no split)", "AS_IS")
        self.splitMode.addItem("Split by materials (recommended)", "MATERIALS")
        self.splitMode.addItem("Split by loose parts", "LOOSE_PARTS")
        fl.addRow("Import:", self.splitMode)

        self.statsLabel = QLabel("—")
        self.statsLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        fl.addRow("Stats:", self.statsLabel)

        root.addWidget(g_paths)

        # ------- Main splitter (left: meshes+render / right: preview)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # ======================
        # LEFT PANEL
        left = QWidget()
        splitter.addWidget(left)
        left_layout = QVBoxLayout(left)

        # Meshes
        g_mesh = QGroupBox("Meshes (drag to reorder layers)")
        v_mesh = QVBoxLayout(g_mesh)

        self.meshTree = QTreeWidget()
        self.meshTree.setColumnCount(2)
        self.meshTree.setHeaderLabels(["Mesh", "Mode"])
        self.meshTree.setDragDropMode(self.meshTree.DragDropMode.InternalMove)
        self.meshTree.setDefaultDropAction(Qt.DropAction.MoveAction)
        v_mesh.addWidget(self.meshTree)

        hb_mesh_btns = QHBoxLayout()
        btn_all_vis = QPushButton("All Visible")
        btn_all_hid = QPushButton("All Hidden")
        btn_all_mask = QPushButton("All Mask")
        btn_all_vis.clicked.connect(lambda: self._set_all_mesh_modes("VISIBLE"))
        btn_all_hid.clicked.connect(lambda: self._set_all_mesh_modes("HIDDEN"))
        btn_all_mask.clicked.connect(lambda: self._set_all_mesh_modes("MASK"))
        hb_mesh_btns.addWidget(btn_all_vis)
        hb_mesh_btns.addWidget(btn_all_hid)
        hb_mesh_btns.addWidget(btn_all_mask)
        hb_mesh_btns.addStretch(1)

        self.chk_per_layer = QCheckBox("Export per-layer spritesheets (layers via holdout from upper meshes)")
        self.chk_per_layer.setChecked(True)
        self.chk_combined = QCheckBox("Export combined spritesheet")
        self.chk_combined.setChecked(True)

        v_mesh.addLayout(hb_mesh_btns)
        v_mesh.addWidget(self.chk_per_layer)
        v_mesh.addWidget(self.chk_combined)

        # Render settings (MOVED HERE)
        g_set = QGroupBox("Render settings")
        fl2 = QFormLayout(g_set)

        self.engine = QComboBox()
        self.engine.addItems(["EEVEE", "CYCLES"])
        fl2.addRow("Engine:", self.engine)

        self.resW = QSpinBox()
        self.resW.setRange(16, 8192)
        self.resW.setValue(512)
        self.resH = QSpinBox()
        self.resH.setRange(16, 8192)
        self.resH.setValue(512)
        hb_res = QHBoxLayout()
        hb_res.addWidget(QLabel("W:"))
        hb_res.addWidget(self.resW)
        hb_res.addWidget(QLabel("H:"))
        hb_res.addWidget(self.resH)
        hb_res.addStretch(1)
        fl2.addRow("Frame size:", hb_res)

        self.samples = QSpinBox()
        self.samples.setRange(1, 4096)
        self.samples.setValue(64)
        fl2.addRow("Cycles samples:", self.samples)

        self.yaw_span = QSpinBox()
        self.yaw_span.setRange(1, 360)
        self.yaw_span.setValue(360)
        fl2.addRow("Yaw span (deg):", self.yaw_span)

        self.angles_count = QSpinBox()
        self.angles_count.setRange(1, 64)
        self.angles_count.setValue(8)
        fl2.addRow("Angles count:", self.angles_count)

        self.animCombo = QComboBox()
        self.animCombo.addItem("(static)")
        fl2.addRow("Animation:", self.animCombo)

        self.animInfoLabel = QLabel("—")
        self.animInfoLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        fl2.addRow("Anim info:", self.animInfoLabel)

        self.fps = QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setValue(24)
        fl2.addRow("FPS:", self.fps)

        self.step = QSpinBox()
        self.step.setRange(1, 240)
        self.step.setValue(1)
        fl2.addRow("Frame step:", self.step)

        self.animCombo.currentTextChanged.connect(self._update_anim_info)
        self.fps.valueChanged.connect(self._update_anim_info)
        self.step.valueChanged.connect(self._update_anim_info)

        # Add to left panel (meshes gets more space)
        left_layout.addWidget(g_mesh, 1)
        left_layout.addWidget(g_set, 0)

        # ======================
        # RIGHT PANEL
        right = QWidget()
        splitter.addWidget(right)
        right_layout = QVBoxLayout(right)

        g_prev = QGroupBox("Preview (camera + lighting)")
        prev_layout = QVBoxLayout(g_prev)

        self.previewImg = PreviewLabel()
        prev_layout.addWidget(self.previewImg, 1)

        # Scroll area for camera + preview + lighting
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_v = QVBoxLayout(scroll_content)
        scroll_v.setContentsMargins(0, 0, 0, 0)

        # Camera
        g_cam = QGroupBox("Camera")
        cam_form = QFormLayout(g_cam)

        self.pitch = QSpinBox()
        self.pitch.setRange(-89, 89)
        self.pitch.setValue(10)
        cam_form.addRow("Pitch (deg):", self.pitch)

        self.yaw_start = QSpinBox()
        self.yaw_start.setRange(0, 359)
        self.yaw_start.setValue(0)
        cam_form.addRow("Yaw (deg):", self.yaw_start)

        self.distanceMult = QDoubleSpinBox()
        self.distanceMult.setRange(0.2, 10.0)
        self.distanceMult.setSingleStep(0.05)
        self.distanceMult.setValue(1.25)
        cam_form.addRow("Distance x:", self.distanceMult)

        hb_off = QHBoxLayout()
        self.offX = QDoubleSpinBox(); self.offX.setRange(-1000, 1000); self.offX.setDecimals(3)
        self.offY = QDoubleSpinBox(); self.offY.setRange(-1000, 1000); self.offY.setDecimals(3)
        self.offZ = QDoubleSpinBox(); self.offZ.setRange(-1000, 1000); self.offZ.setDecimals(3)
        hb_off.addWidget(QLabel("X")); hb_off.addWidget(self.offX)
        hb_off.addWidget(QLabel("Y")); hb_off.addWidget(self.offY)
        hb_off.addWidget(QLabel("Z")); hb_off.addWidget(self.offZ)
        cam_form.addRow("Target offset:", hb_off)

        self.fov = QSpinBox()
        self.fov.setRange(10, 120)
        self.fov.setValue(50)
        cam_form.addRow("FOV (deg):", self.fov)

        self.roll = QSpinBox()
        self.roll.setRange(-180, 180)
        self.roll.setValue(0)
        cam_form.addRow("Roll (deg):", self.roll)

        self.ortho = QCheckBox("Orthographic")
        cam_form.addRow("", self.ortho)

        self.orthoScale = QDoubleSpinBox()
        self.orthoScale.setRange(0.001, 10000)
        self.orthoScale.setDecimals(3)
        self.orthoScale.setValue(3.0)
        cam_form.addRow("Ortho scale:", self.orthoScale)

        # Preview settings
        g_prevset = QGroupBox("Preview settings")
        prevset = QFormLayout(g_prevset)

        self.previewW = QSpinBox()
        self.previewW.setRange(64, 2048)
        self.previewW.setValue(512)
        self.previewH = QSpinBox()
        self.previewH.setRange(64, 2048)
        self.previewH.setValue(512)
        hb_ps = QHBoxLayout()
        hb_ps.addWidget(QLabel("W:")); hb_ps.addWidget(self.previewW)
        hb_ps.addWidget(QLabel("H:")); hb_ps.addWidget(self.previewH)
        hb_ps.addStretch(1)
        prevset.addRow("Preview size:", hb_ps)

        self.posePercent = QSpinBox()
        self.posePercent.setRange(0, 100)
        self.posePercent.setValue(0)
        prevset.addRow("Pose %:", self.posePercent)

        self.previewSamples = QSpinBox()
        self.previewSamples.setRange(1, 256)
        self.previewSamples.setValue(16)
        prevset.addRow("Samples:", self.previewSamples)

        self.previewHoldouts = QCheckBox("Preview holdouts (MASK meshes cut alpha)")
        self.previewHoldouts.setChecked(True)
        prevset.addRow("", self.previewHoldouts)

        # Lighting
        g_light = QGroupBox("Lighting")
        light_form = QFormLayout(g_light)

        self.lightPreset = QComboBox()
        self.lightPreset.addItem("Studio (recommended)", "STUDIO")
        self.lightPreset.addItem("Flat (very soft shadows)", "FLAT")
        self.lightPreset.addItem("Dramatic (strong key light)", "DRAMATIC")
        light_form.addRow("Preset:", self.lightPreset)

        self.lightStrength = QDoubleSpinBox()
        self.lightStrength.setRange(0.05, 5.0)
        self.lightStrength.setSingleStep(0.05)
        self.lightStrength.setValue(1.0)
        light_form.addRow("Strength:", self.lightStrength)

        self.worldStrength = QDoubleSpinBox()
        self.worldStrength.setRange(0.0, 3.0)
        self.worldStrength.setSingleStep(0.05)
        self.worldStrength.setValue(0.35)
        light_form.addRow("Ambient (world):", self.worldStrength)

        scroll_v.addWidget(g_cam)
        scroll_v.addWidget(g_prevset)
        scroll_v.addWidget(g_light)
        scroll_v.addStretch(1)

        scroll.setWidget(scroll_content)

        # Чуть больше места под настройки, чтобы чаще помещались без скролла
        scroll.setMaximumHeight(520)

        prev_layout.addWidget(scroll, 0)

        hb_prev_btns = QHBoxLayout()
        self.btn_load = QPushButton("Load .glb")
        self.btn_load.clicked.connect(self.load_glb)

        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self.preview)

        self.btn_render = QPushButton("Render")
        self.btn_render.clicked.connect(self.render)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_process)
        hb_prev_btns.addWidget(self.btn_cancel)

        
        hb_prev_btns.addWidget(self.btn_load)
        hb_prev_btns.addWidget(self.btn_preview)
        hb_prev_btns.addWidget(self.btn_render)
        hb_prev_btns.addStretch(1)
        prev_layout.addLayout(hb_prev_btns)

        right_layout.addWidget(g_prev, 1)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # ------- progress + log (делаем лог невысоким, чтобы не отъедал место у превью)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        root.addWidget(self.log)

        # menu
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        m = self.menuBar().addMenu("File")
        m.addAction(act_quit)

    # ---------- UI helpers ----------
    def cancel_process(self) -> None:
        if self.proc.state() == QProcess.ProcessState.NotRunning:
            return
        self._user_canceled = True
        self._append_log("\nCancel requested. Stopping Blender…")
        self.proc.kill()


    def _browse_blender(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Blender executable")
        if path:
            self.blenderPath.setText(path)

    def _browse_glb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select .glb file", filter="glTF Binary (*.glb)")
        if path:
            self.glbPath.setText(path)

    def _browse_outdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.outDir.setText(path)

    def _append_log(self, text: str) -> None:
        self.log.append(text.rstrip("\n"))

    def _error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg)

    def _set_enabled(self, enabled: bool) -> None:
        widgets = [
            self.blenderPath, self.glbPath, self.outDir, self.splitMode,
            self.btn_load, self.btn_preview, self.btn_render,
            self.meshTree, self.chk_per_layer, self.chk_combined,
            self.engine, self.resW, self.resH, self.samples,
            self.pitch, self.yaw_start, self.distanceMult,
            self.offX, self.offY, self.offZ, self.fov, self.roll,
            self.ortho, self.orthoScale,
            self.previewW, self.previewH, self.posePercent, self.previewSamples, self.previewHoldouts,
            self.lightPreset, self.lightStrength, self.worldStrength,
            self.yaw_span, self.angles_count,
            self.animCombo, self.fps, self.step,
        ]
        for w in widgets:
            w.setEnabled(enabled)
        # Cancel должен быть доступен когда UI заблокирован
        if hasattr(self, "btn_cancel"):
            self.btn_cancel.setEnabled(not enabled)


    def _set_all_mesh_modes(self, mode: str) -> None:
        for i in range(self.meshTree.topLevelItemCount()):
            it = self.meshTree.topLevelItem(i)
            combo = self.meshTree.itemWidget(it, 1)
            if isinstance(combo, QComboBox):
                combo.setCurrentText(mode)

    def _update_anim_info(self) -> None:
        name = self.animCombo.currentText()
        step = max(1, int(self.step.value()))
        fps = max(1, int(self.fps.value()))

        if not name or name == "(static)":
            self.animInfoLabel.setText("Static (1 frame)")
            return

        info = self.anim_meta.get(name)

        if isinstance(info, dict) and "frame_start" in info and "frame_end" in info:
            fs = int(round(float(info["frame_start"])))
            fe = int(round(float(info["frame_end"])))
            if fe < fs:
                fs, fe = fe, fs

            source_frames = max(1, fe - fs + 1)
            rendered_frames = ((source_frames - 1) // step) + 1

            dur_src = source_frames / float(fps)
            dur_rnd = rendered_frames / float(fps)

            self.animInfoLabel.setText(
                f"range {fs}–{fe} | source: {source_frames} frames ({dur_src:.2f}s @ {fps}fps) | "
                f"step {step} ⇒ {rendered_frames} frames ({dur_rnd:.2f}s)"
            )
        else:
            self.animInfoLabel.setText(f"{name} | range unknown | step {step}")

    # ---------- Process ----------
    def _start_process(self, program: str, args: list[str], stage: str) -> None:
        self.stage = stage
        self._user_canceled = False
        # "занято" пока не увидим ###PROGRESS
        self.progress.setRange(0, 0)
        self.progress.setValue(0)
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._append_log(f"\n== {stage.upper()} ==\n{program} {' '.join(args)}")
        self.proc.setProgram(program)
        self.proc.setArguments(args)
        self.proc.start()
        if not self.proc.waitForStarted(5000):
            self.stage = "idle"
            self._set_enabled(True)
            self._error("Failed to start", "Could not start process (check Blender path).")

    def _on_stdout(self) -> None:
        chunk = self.proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._stdout_buffer += chunk

        for line in chunk.splitlines():
            s = line.strip()
            if not s:
                continue

            if s.startswith("###PROGRESS"):
                try:
                    p = float(s.split()[1])
                    # если был "busy" — переводим в нормальный
                    if self.progress.maximum() == 0 and self.progress.minimum() == 0:
                        self.progress.setRange(0, 1000)
                    self.progress.setValue(max(0, min(1000, int(p * 1000))))
                except Exception:
                    pass
            else:
                self._append_log(s)


    def _on_stderr(self) -> None:
        chunk = self.proc.readAllStandardError().data().decode("utf-8", errors="replace")
        self._stderr_buffer += chunk
        for line in chunk.splitlines():
            if line.strip():
                self._append_log("[stderr] " + line)

    def _on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._user_canceled:
            self._user_canceled = False
            self.stage = "idle"
            self._set_enabled(True)
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)
            self._append_log("Canceled.")
            return

        if status != QProcess.ExitStatus.NormalExit or code != 0:
            self._set_enabled(True)
            self.stage = "idle"
            self._error("Process error", f"Failed (code={code}).\n\n{self._stderr_buffer[-4000:]}")
            return

        if self.stage == "probing":
            self._handle_probe_finished()
        elif self.stage == "previewing":
            self._handle_preview_finished()
        elif self.stage == "rendering":
            self._handle_render_finished()
        elif self.stage == "packing":
            self._handle_pack_finished()

    # ---------- Steps ----------
    def load_glb(self) -> None:
        blender = self.blenderPath.text().strip()
        glb = self.glbPath.text().strip()
        if not blender or not Path(blender).exists():
            self._error("Blender not found", "Set correct Blender executable path.")
            return
        if not glb or not Path(glb).exists():
            self._error("GLB not found", "Select a .glb file.")
            return

        self._set_enabled(False)
        self.progress.setValue(0)

        script = str(Path(__file__).with_name("blender_probe.py"))
        split_mode = self.splitMode.currentData()

        args = [
            "--background", "--factory-startup",
            "--python", script,
            "--", "--input", glb, "--split_mode", split_mode,
        ]
        self._start_process(blender, args, "probing")

    def _handle_probe_finished(self) -> None:
        text = self._stdout_buffer
        begin = text.find("###BEGIN_JSON")
        end = text.find("###END_JSON")
        if begin == -1 or end == -1 or end <= begin:
            self._set_enabled(True)
            self.stage = "idle"
            self._error("Probe failed", "Could not parse Blender output (no JSON markers).")
            return

        payload = text[begin + len("###BEGIN_JSON"):end].strip()
        try:
            data = json.loads(payload)
        except Exception as e:
            self._set_enabled(True)
            self.stage = "idle"
            self._error("Probe failed", f"Invalid JSON: {e}")
            return

        meshes = data.get("meshes", [])
        details = {d["name"]: d for d in data.get("mesh_details", [])}
        anims = data.get("animations", [])
        stats = data.get("stats", {})

        self.meshTree.clear()
        for name in meshes:
            it = QTreeWidgetItem([name, ""])
            det = details.get(name)
            if det:
                tip = f"verts={det.get('verts')} polys={det.get('polys')}\nmaterials:\n- " + "\n- ".join(det.get("materials", []))
                it.setToolTip(0, tip)
            self.meshTree.addTopLevelItem(it)

            combo = QComboBox()
            combo.addItems(["VISIBLE", "HIDDEN", "MASK"])
            combo.setCurrentText("VISIBLE")
            self.meshTree.setItemWidget(it, 1, combo)

        # Animations + meta
        self.anim_meta = {}
        self.animCombo.clear()
        self.animCombo.addItem("(static)")
        for a in anims:
            if isinstance(a, dict):
                name = a.get("name", "")
                if name:
                    self.animCombo.addItem(name)
                    self.anim_meta[name] = a
            else:
                name = str(a)
                if name:
                    self.animCombo.addItem(name)
                    self.anim_meta[name] = None

        self.statsLabel.setText(
            f"objects={stats.get('objects_total','?')} "
            f"meshes={stats.get('mesh_objects','?')} "
            f"armatures={stats.get('armatures','?')} "
            f"materials={stats.get('materials','?')} "
            f"images={stats.get('images','?')} "
            f"split={stats.get('split_mode','?')}"
        )

        self._append_log(f"\nLoaded meshes: {len(meshes)}; animations: {len(anims)}")
        self._update_anim_info()

        self.stage = "idle"
        self._set_enabled(True)
        self.progress.setValue(0)

    def preview(self) -> None:
        blender = self.blenderPath.text().strip()
        glb = self.glbPath.text().strip()
        out_dir = self.outDir.text().strip()

        if not blender or not Path(blender).exists():
            self._error("Blender not found", "Set correct Blender executable path.")
            return
        if not glb or not Path(glb).exists():
            self._error("GLB not found", "Select a .glb file.")
            return

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self._preview_path = str(Path(out_dir) / "_preview.png")

        config = self._build_config(glb, out_dir)
        config["preview"] = {
            "out_path": self._preview_path,
            "resolution": [int(self.previewW.value()), int(self.previewH.value())],
            "pose_percent": int(self.posePercent.value()),
            "samples": int(self.previewSamples.value()),
            "yaw": float(self.yaw_start.value()),
            "pitch": float(self.pitch.value()),
            "preview_holdouts": bool(self.previewHoldouts.isChecked()),
        }

        config_path = Path(out_dir) / "config_preview.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        self._set_enabled(False)
        self.progress.setValue(0)

        script = str(Path(__file__).with_name("blender_preview.py"))
        args = [
            "--background", "--factory-startup",
            "--python", script,
            "--", "--config", str(config_path),
        ]
        self._start_process(blender, args, "previewing")

    def _handle_preview_finished(self) -> None:
        self.stage = "idle"
        self._set_enabled(True)
        self.progress.setValue(1000)

        if not self._preview_path or not Path(self._preview_path).exists():
            self._append_log("Preview done, but preview image not found.")
            return

        pix = QPixmap(self._preview_path)
        if pix.isNull():
            self._append_log("Preview image failed to load.")
            return

        self.previewImg.set_preview_pixmap(pix)

    def render(self) -> None:
        blender = self.blenderPath.text().strip()
        glb = self.glbPath.text().strip()
        out_dir = self.outDir.text().strip()

        if not blender or not Path(blender).exists():
            self._error("Blender not found", "Set correct Blender executable path.")
            return
        if not glb or not Path(glb).exists():
            self._error("GLB not found", "Select a .glb file.")
            return

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        config = self._build_config(glb, out_dir)
        config_path = Path(out_dir) / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        self._set_enabled(False)
        self.progress.setValue(0)

        script = str(Path(__file__).with_name("blender_render.py"))
        args = [
            "--background", "--factory-startup",
            "--python", script,
            "--", "--config", str(config_path),
        ]
        self._start_process(blender, args, "rendering")

    def _handle_render_finished(self) -> None:
        out_dir = self.outDir.text().strip()
        script = str(Path(__file__).with_name("pack_sheets.py"))
        python_exe = sys.executable

        self.progress.setValue(0)
        args = ["-u", script, "--output_dir", out_dir]
        self._start_process(python_exe, args, "packing")

    def _handle_pack_finished(self) -> None:
        self.stage = "idle"
        self._set_enabled(True)
        self.progress.setValue(1000)
        self._append_log("\n✅ Done. Check output folder.")

    # ---------- Config helpers ----------
    def _collect_layers(self) -> list[dict]:
        layers = []
        for i in range(self.meshTree.topLevelItemCount()):
            it = self.meshTree.topLevelItem(i)
            name = it.text(0)
            combo = self.meshTree.itemWidget(it, 1)
            mode = "VISIBLE"
            if isinstance(combo, QComboBox):
                mode = combo.currentText()
            layers.append({"name": name, "mode": mode, "order": i})
        return layers

    def _compute_angles(self) -> list[dict]:
        pitch = float(self.pitch.value())
        start = float(self.yaw_start.value())
        span = float(self.yaw_span.value())
        count = int(self.angles_count.value())

        if count <= 1:
            yaws = [start]
        else:
            step = span / count
            yaws = [start + i * step for i in range(count)]

        return [{"yaw": float(y % 360.0), "pitch": pitch} for y in yaws]

    def _build_config(self, glb: str, out_dir: str) -> dict:
        angles = self._compute_angles()
        layers = self._collect_layers()

        config = {
            "input_glb": glb,
            "output_dir": out_dir,
            "import": {"split_mode": self.splitMode.currentData()},
            "engine": self.engine.currentText(),
            "resolution": [int(self.resW.value()), int(self.resH.value())],
            "cycles_samples": int(self.samples.value()),
            "angles": angles,
            "camera": {
                "distance_mult": float(self.distanceMult.value()),
                "target_offset": [float(self.offX.value()), float(self.offY.value()), float(self.offZ.value())],
                "fov_deg": float(self.fov.value()),
                "roll_deg": float(self.roll.value()),
                "ortho": bool(self.ortho.isChecked()),
                "ortho_scale": float(self.orthoScale.value()),
            },
            "lighting": {
                "preset": self.lightPreset.currentData(),
                "strength": float(self.lightStrength.value()),
                "world_strength": float(self.worldStrength.value()),
            },
            "animation": {
                "name": self.animCombo.currentText(),
                "fps": int(self.fps.value()),
                "step": int(self.step.value()),
            },
            "export": {
                "per_layer": bool(self.chk_per_layer.isChecked()),
                "combined": bool(self.chk_combined.isChecked()),
            },
            "layers": layers,
        }
        return config


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1280, 900)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
