"""
Autonomous Car — Pi Server
==========================
Run on the Raspberry Pi:
    cd ~/licenta
    python app.py

Open  http://<PI_IP>:5000  in any browser on the same WiFi network.

Optional env vars:
    CAR_SECRET    – Flask secret key
    CAR_USERNAME  – login username   (default: admin)
    CAR_PASSWORD  – login password   (default: admin123)
    CAMERA_INDEX  – cv2 camera index (default: 0)
"""

import csv
import hmac
import io
import json
import os
import queue
import shutil
import sys
import threading
import time
from datetime import datetime, timedelta
from functools import wraps

import cv2
import numpy as np
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_socketio import SocketIO, emit

try:
    import psutil as _psutil
except ImportError:
    _psutil = None
    print("[server] psutil not found — CPU/RAM stats disabled  (pip install psutil)")

sys.path.insert(0, os.path.dirname(__file__))
from MotorModule import MotorModule
import utlis
from sign_detection import SignDetector, DriveBehavior, annotate_frame
from obstacle_detection import ObstacleDetector
from utlis import thresholding_robust, thresholding_course, thresholding_road, get_lane_center_from_lines
from DS4Module import DS4Controller
from pid import PIDController
from parking_module import ParkingController

try:
    from UltrasonicModule import get_array
    _ultra = get_array()           # 6× HC-SR04 (3 față, 3 spate)
    print(f"[ultrasonic] array ready — {_ultra.count}/6 senzori activi")
    if _ultra.count < 6:
        print("[ultrasonic] ⚠ unii senzori lipsesc — verifică cablajul/pinii în log")
except Exception as _ue:
    _ultra = None
    print(f"[ultrasonic] array not available ({_ue}) — crash & parking disabled")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SECRET_KEY   = os.getenv("CAR_SECRET",   "change-me-in-production")
USERNAME     = os.getenv("CAR_USERNAME", "admin")
PASSWORD     = os.getenv("CAR_PASSWORD", "admin123")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

HOST = "0.0.0.0"
PORT = 5000

# Driving
BASE_SPEED = 0.32
BASE_TURN  = 1.0
MAX_TURN   = 1.0

# Bird's-eye warp  [widthTop, heightTop, widthBottom, heightBottom]
# Tune these live via the browser's "Lane Calibration" panel,
# or run  python LaneModule.py  on a machine with a display.
WARP_POINTS = [65, 105, 92, 240]
_warp_lock  = threading.Lock()

# Threading rates
STREAM_FPS  = 30
LANE_FPS    = 20
SIGN_FPS    = 8     # sign detection is heavier — run at 8 Hz
RECORD_FPS  = 10    # frames saved per second during recording

# Paths
_BASE        = os.path.dirname(__file__)
DATASET_DIR  = os.path.join(_BASE, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Flask + SocketIO
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"]              = SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ══════════════════════════════════════════════════════════════════════════════
# Hardware
# ══════════════════════════════════════════════════════════════════════════════

motor      = MotorModule()
motor_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
# Shared state
# ══════════════════════════════════════════════════════════════════════════════

state_lock    = threading.Lock()
manual_mode   = True
active_keys   = set()
current_speed = BASE_SPEED
lane_curve        = 0.0
_lane_curve_smooth = 0.0   # EMA-smoothed lane curve (warp mode) — anti-spazz
_lane_raw_hist: list = []  # ultimele câteva curbe brute → filtru median

# ── Lane smoothing tuning (warp mode) ──────────────────────────────────────────
LANE_SMOOTH_ALPHA = 0.65   # 0=fără filtru, →1 mai neted dar mai cu întârziere
LANE_MAX_STEP     = 0.18   # schimbarea maximă de curbă pe cadru (anti-salt)
LANE_MEDIAN_N     = 3      # fereastră median → respinge complet pâlpâirile de 1 cadru

# Camera buffers
raw_lock              = threading.Lock()
annotated_lock        = threading.Lock()
latest_raw_frame      = None
latest_annotated_frame = None
_curve_list: list     = []

# Camera object (kept for reference; properties set via software, not cap.set)
_cap      = None
_cap_lock = threading.Lock()

# Software camera adjustments (applied per-frame, no hardware calls)
_cam_lock       = threading.Lock()
_cam_brightness = 0      # offset added to every pixel:  -100 … +100  (0 = off)
_cam_contrast   = 1.0    # multiplier on every pixel:    0.0  … 2.0   (1.0 = off)
_cam_saturation = 1.0    # HSV S-channel multiplier:     0.0  … 2.0   (1.0 = off)


def _apply_cam_adjust(frame: np.ndarray) -> np.ndarray:
    """Apply software brightness / contrast / saturation to a BGR frame."""
    with _cam_lock:
        b, c, s = _cam_brightness, _cam_contrast, _cam_saturation
    if b == 0 and c == 1.0 and s == 1.0:
        return frame                       # fast path — nothing to do
    # Brightness + contrast  (convertScaleAbs clips to 0-255 safely)
    out = cv2.convertScaleAbs(frame, alpha=c, beta=b)
    # Saturation via HSV
    if s != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

# Sign detection
sign_detector    = SignDetector()           # loads yolov8n.pt on first use
drive_behavior   = DriveBehavior()
signs_enabled    = False
latest_detections: list = []
detections_lock  = threading.Lock()

_calib_overlay   = False   # draw warp trapezoid on live feed

# Mock lane mode: ignore camera, use a manually-set curve value
mock_lane_mode  = False
mock_lane_curve = 0.0      # set by the UI slider, range [-1, 1]

# Lane detection mode: "warp" (bird's eye) | "direct" (sliding window, no warp)
lane_mode    = "warp"
roi_top_frac = 0.55   # direct mode: ignore top fraction of frame (0=whole frame, 0.7=bottom 30%)

# Behavioral cloning inference
behavioral_mode    = False
_behavioral_driver = None    # BehavioralDriver instance (loaded on demand)
behavioral_lock    = threading.Lock()

# Obstacle detection
obstacle_detector  = ObstacleDetector()
obstacles_enabled  = False
latest_obstacle    = {"detected": False, "bbox": None, "area_frac": 0.0, "distance_frac": 0.0}
obstacle_lock      = threading.Lock()
_obstacle_stopped  = False   # True when motors halted due to an obstacle

# Ultrasonic crash detection (uses the 3 FRONT sensors while driving forward)
CRASH_DISTANCE_CM    = 30.0  # brake threshold in cm
ultrasonic_enabled   = False
ultrasonic_cm        = None   # closest front reading, or None
ultrasonic_all       = {}     # latest dict of all 6 sensors {name: cm|None}
_ultrasonic_stopped  = False  # True when motors halted by ultrasonic sensor
ultrasonic_lock      = threading.Lock()

# ── Auto-parking (reverse into a 3-wall garage) ─────────────────────────────────
def _parking_status_cb(status: dict):
    socketio.emit("parking_status", status)

if _ultra is not None:
    parking = ParkingController(
        motor, motor_lock,
        get_distances=_ultra.get_distances,
        on_status=_parking_status_cb,
    )
else:
    parking = None


def _parking_active() -> bool:
    """True when the auto-parking maneuver owns the motors."""
    return parking is not None and parking.active

# DS4 / PS4 Controller
ds4          = DS4Controller()
ds4_enabled  = True    # user can toggle from UI (True = use DS4 when connected)

# PID Controller for lane-keeping
pid              = PIDController(kp=0.45, ki=0.01, kd=0.10)
pid_enabled      = True     # ON by default — provides derivative damping to kill oscillation

# Adaptive Speed Control — reduces speed proportionally to curve²
adaptive_speed_enabled = True
curve_k                = 0.5   # 0=no reduction  1=full reduction at max curve

# ── Speed Limit Enforcement ────────────────────────────────────────────────────
# Map GTSRB label → VITEZĂ ȚINTĂ ABSOLUTĂ (fracție motor 0-1) la care merge mașina.
# Semne fizice pe pistă: STOP, CEDEAZĂ, 30, 50, 90, 100. GTSRB n-are "90" → 80 = 90.
#
# IMPORTANT: acestea sunt viteze ABSOLUTE, nu plafoane. Când un semn e activ,
# mașina merge LA această viteză (înlocuiește slider-ul), ca să se VADĂ că ascultă
# de semn. Valorile trebuie să fie peste pragul de pornire al motoarelor (~0.38)
# și clar diferite între ele. Reglează dacă 30 se oprește sau 100 e prea rapid.
SPEED_LIMIT_MAP = {
    "speed_limit_30":      0.40,   # cel mai lent, dar încă se mișcă
    "speed_limit_50":      0.47,
    "speed_limit_80":      0.53,   # remapat ca 90 km/h (≈ viteza normală de pistă)
    "speed_limit_100":     0.58,   # cel mai rapid (redus de la 0.66)
    "end_speed_limit_80":  None,   # None = anulează limita activă
    "end_restrictions":    None,
}

# Afișare km/h pentru fiecare clasă GTSRB → eticheta corectă pe UI
SPEED_LIMIT_DISPLAY = {
    "speed_limit_30":  "30",
    "speed_limit_50":  "50",
    "speed_limit_80":  "90",    # remapat
    "speed_limit_100": "100",
}
active_speed_limit_frac = None   # None = no active limit
speed_limit_enabled     = True   # user can disable enforcement from UI
speed_limit_lock        = threading.Lock()


def _cap_to_speed_limit(speed: float) -> float:
    """Limitează viteza ÎNAINTE la limita activă din semne (moduri manuale)."""
    with speed_limit_lock:
        sl = active_speed_limit_frac if speed_limit_enabled else None
    if sl is not None and speed > sl:
        return sl
    return speed

# ── Lane Detection Confidence ──────────────────────────────────────────────────
lane_confidence = 0   # 0-100%; updated by lane thread; -1 = N/A (mock/behavioral)

# ── Config Manager ─────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(_BASE, "car_config.json")

DEFAULT_CONFIG = {
    "base_speed":   0.32,
    "base_turn":    1.0,
    "warp_points":  [119, 105, 138, 240],
    "pid_kp":       0.35,
    "pid_ki":       0.01,
    "pid_kd":       0.12,
    "pid_enabled":  True,
    "curve_k":      0.5,
    "adaptive_speed": True,
    "roi_top_frac": 0.55,
    "lane_mode":    "warp",
    "lane_fps":     20,
    "sign_fps":     8,
    "record_fps":   10,
}

CONFIG_PRESETS = {
    "Default":     {**DEFAULT_CONFIG},
    "Pistă Mică":  {**DEFAULT_CONFIG, "lane_mode": "direct", "roi_top_frac": 0.55,
                    "base_speed": 0.40, "curve_k": 0.5, "pid_kp": 0.30, "pid_kd": 0.25,
                    "pid_enabled": True, "adaptive_speed": True},
    "Pistă Mare":  {**DEFAULT_CONFIG, "lane_mode": "warp",
                    "base_speed": 0.65, "curve_k": 0.4, "pid_kp": 0.40, "pid_kd": 0.30,
                    "pid_enabled": True, "adaptive_speed": True},
    "Demo Lent":   {**DEFAULT_CONFIG, "base_speed": 0.28, "curve_k": 0.4, "pid_kp": 0.25,
                    "pid_kd": 0.20, "pid_enabled": True, "adaptive_speed": True},
}


def _load_config_file() -> dict:
    """Load saved config from disk, falling back to defaults."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def _apply_config(cfg: dict):
    """Apply a config dict to the running state immediately."""
    global BASE_SPEED, current_speed, curve_k, roi_top_frac, lane_mode, _curve_list, WARP_POINTS
    global pid_enabled, adaptive_speed_enabled
    BASE_SPEED    = float(cfg.get("base_speed",   BASE_SPEED))
    with state_lock:
        current_speed = BASE_SPEED
        curve_k       = float(cfg.get("curve_k",      curve_k))
        roi_top_frac  = float(cfg.get("roi_top_frac", roi_top_frac))
        new_lm        = cfg.get("lane_mode", lane_mode)
        if new_lm in ("warp", "direct"):
            lane_mode = new_lm
        if "pid_enabled" in cfg:
            pid_enabled = bool(cfg["pid_enabled"])
        if "adaptive_speed" in cfg:
            adaptive_speed_enabled = bool(cfg["adaptive_speed"])
        _curve_list.clear()
    wp = cfg.get("warp_points")
    if isinstance(wp, list) and len(wp) == 4:
        with _warp_lock:
            WARP_POINTS = [int(x) for x in wp]
    gs = pid.get_state()
    pid.set_gains(
        kp=float(cfg.get("pid_kp", gs["kp"])),
        ki=float(cfg.get("pid_ki", gs["ki"])),
        kd=float(cfg.get("pid_kd", gs["kd"])),
    )
    pid.reset()
    print(f"[config] applied: {cfg}")


# Aplică config-ul salvat imediat la pornire (funcționează și cu systemd)
_apply_config(_load_config_file())

# ── Race Timer ─────────────────────────────────────────────────────────────────
race_lock       = threading.Lock()
race_active     = False
race_start_time: float = None
race_laps: list = []        # lap durations in seconds
race_lap_start: float = None


def _get_race_state() -> dict:
    with race_lock:
        elapsed     = round(time.time() - race_start_time, 2) if race_active and race_start_time else 0.0
        lap_elapsed = round(time.time() - race_lap_start,  2) if race_active and race_lap_start  else 0.0
        laps        = list(race_laps)
    return {
        "active":      race_active,
        "elapsed":     elapsed,
        "lap_elapsed": lap_elapsed,
        "laps":        laps,
        "lap_count":   len(laps),
        "best_lap":    round(min(laps), 3) if laps else None,
    }


def _race_timer_thread():
    while True:
        time.sleep(0.1)
        with race_lock:
            active = race_active
        if active:
            socketio.emit("race_tick", _get_race_state())


threading.Thread(target=_race_timer_thread, daemon=True).start()


# ── Video Export ───────────────────────────────────────────────────────────────
_export_jobs  = {}    # session_id → {"status","progress","path","error"}
_export_lock2 = threading.Lock()


def _export_session_thread(session_id: str):
    """Compile session frames into an MP4 with telemetry overlay."""
    sess_dir  = os.path.join(DATASET_DIR, session_id)
    img_dir   = os.path.join(sess_dir, "images")
    csv_path  = os.path.join(sess_dir, "labels.csv")
    out_path  = os.path.join(sess_dir, "export.mp4")

    try:
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        if not frames:
            raise ValueError("Sesiunea nu are cadre")

        # Load CSV labels
        meta: dict = {}
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    meta[row["filename"]] = row

        first = cv2.imread(os.path.join(img_dir, frames[0]))
        if first is None:
            raise ValueError("Nu pot citi primul cadru")
        h, w = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, RECORD_FPS, (w, h))

        n = len(frames)
        for i, fname in enumerate(frames):
            frame = cv2.imread(os.path.join(img_dir, fname))
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)

            row = meta.get(fname, {})
            spd = float(row.get("speed", 0))
            trn = float(row.get("turn",  0))
            ts  = row.get("timestamp", "")

            # Telemetry overlay bar at bottom
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 38), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            text = f"Speed:{spd:+.2f}  Turn:{trn:+.2f}  #{i+1:04d}/{n}  {ts}"
            cv2.putText(frame, text, (8, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 255), 1)
            writer.write(frame)

            progress = round((i + 1) / n * 100)
            with _export_lock2:
                _export_jobs[session_id]["progress"] = progress
            if progress % 5 == 0 or i == n - 1:
                socketio.emit("export_progress", {
                    "session": session_id, "progress": progress, "done": False})

        writer.release()

        with _export_lock2:
            _export_jobs[session_id].update({"status": "done", "progress": 100, "path": out_path})

        socketio.emit("export_progress", {"session": session_id, "progress": 100, "done": True})
        print(f"[export] done → {out_path}")

    except Exception as exc:
        with _export_lock2:
            _export_jobs[session_id].update({"status": "error", "error": str(exc)})
        socketio.emit("export_progress", {"session": session_id, "error": str(exc)})
        print(f"[export] error: {exc}")

# Recording
rec_lock      = threading.Lock()
recording     = False
rec_frames    = 0                          # total frames saved this session
rec_session   = ""                         # current session folder name
_rec_queue    = queue.Queue(maxsize=600)   # (frame, speed, turn, index)

# ── Motor Playback (Record & Replay) — BAZAT PE EVENIMENTE ───────────────────
# Fiecare comandă de motor (tastatură / joystick / DS4) se salvează ca
# (durată_ținută_secunde, viteză, viraj), cu timp EXACT (time.monotonic()).
# Astfel fiecare segment ține exact cât a fost apăsat butonul, iar redarea
# reproduce mișcarea fidel (deadline absolut la playback → fără drift).
_pb_lock           = threading.Lock()
_pb_recording      = False                 # înregistrare activă
_pb_sequence: list = []                    # [(durată_s, speed, turn), ...] secvența finală
_pb_rec_events: list = []                  # buffer de segmente în timpul înregistrării
_pb_rec_last       = None                  # (t_monotonic, speed, turn) ultima comandă
_pb_playing        = False                 # playback activ
_pb_current_turn   = 0.0                   # virajul curent din playback (pentru arrow)

def pb_start_record():
    """Începe înregistrarea comenzilor de motor (event-based, timp exact)."""
    global _pb_recording, _pb_rec_events, _pb_rec_last
    with _pb_lock:
        _pb_recording  = True
        _pb_rec_events = []
        _pb_rec_last   = (time.monotonic(), 0.0, 0.0)   # mașina pleacă din repaus
    print("[playback] ● RECORDING started (event-based)")
    socketio.emit("pb_state", {"recording": True, "has_sequence": bool(_pb_sequence)})

def pb_record_command(spd: float, trn: float):
    """
    Înregistrează o comandă de motor cu durată exactă.
    Apelată din TOATE sursele manuale (tastatură, joystick, DS4) imediat după
    motor.move(). Închide segmentul comenzii anterioare cu durata reală pentru
    care a fost ținută, apoi deschide unul nou.
    """
    global _pb_rec_last
    with _pb_lock:
        if not _pb_recording:
            return
        spd = round(float(spd), 3)
        trn = round(float(trn), 3)
        t   = time.monotonic()
        if _pb_rec_last is not None:
            t_prev, s_prev, tr_prev = _pb_rec_last
            if spd == s_prev and trn == tr_prev:
                return                       # aceeași comandă → segmentul se extinde
            dur = t - t_prev
            if dur > 0.0:
                _pb_rec_events.append((dur, s_prev, tr_prev))
        _pb_rec_last = (t, spd, trn)

def pb_stop_record():
    """Oprește înregistrarea, închide ultimul segment, salvează secvența."""
    global _pb_recording, _pb_sequence, _pb_rec_last
    with _pb_lock:
        _pb_recording = False
        if _pb_rec_last is not None:
            t_prev, s_prev, tr_prev = _pb_rec_last
            dur = time.monotonic() - t_prev
            if dur > 0.0:
                _pb_rec_events.append((dur, s_prev, tr_prev))
        _pb_sequence = list(_pb_rec_events)
        _pb_rec_last = None
        n     = len(_pb_sequence)
        total = sum(d for d, _, _ in _pb_sequence)
    print(f"[playback] ■ RECORDING stopped — {n} segmente, {total:.2f}s")
    socketio.emit("pb_state", {"recording": False, "has_sequence": n > 0,
                               "duration": round(total, 1), "count": n})

def pb_start_play():
    """Pornește playback-ul în buclă infinită."""
    global _pb_playing
    with _pb_lock:
        if not _pb_sequence:
            print("[playback] ▶ no sequence to play")
            return
        _pb_playing = True
    threading.Thread(target=_pb_play_thread, daemon=True).start()
    print("[playback] ▶ PLAYBACK started")
    socketio.emit("pb_state", {"playing": True})

def pb_stop_play():
    """Oprește playback-ul."""
    global _pb_playing
    with _pb_lock:
        _pb_playing = False
    with motor_lock:
        motor.stop()
    print("[playback] ■ PLAYBACK stopped")
    socketio.emit("pb_state", {"playing": False})

def _pb_play_thread():
    """
    Reproduce secvența în buclă infinită, cu timing EXACT.
    Folosește un deadline absolut (start + durată_cumulată) ca durata totală să
    fie identică cu înregistrarea — fără drift acumulat din time.sleep.
    """
    global _pb_playing, _pb_current_turn
    try:
        while True:
            with _pb_lock:
                if not _pb_playing or not _pb_sequence:
                    return
                seq = list(_pb_sequence)
            start   = time.monotonic()
            elapsed = 0.0
            for dur, spd, trn in seq:
                with _pb_lock:
                    if not _pb_playing:
                        return
                _pb_current_turn = trn          # virajul curent pentru arrow
                with motor_lock:
                    motor.move(spd, trn)
                elapsed += dur
                remaining = (start + elapsed) - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            # secvența completă → bucla while reia de la capăt
    finally:
        with motor_lock:
            motor.stop()
        _pb_current_turn = 0.0
        with _pb_lock:
            _pb_playing = False
        socketio.emit("pb_state", {"playing": False})

# ══════════════════════════════════════════════════════════════════════════════
# Camera capture thread
# ══════════════════════════════════════════════════════════════════════════════

def _camera_thread():
    global latest_raw_frame, _cap

    # Try picamera2 first (Raspberry Pi Camera Module on Bookworm/libcamera)
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        sensor_size = picam2.camera_properties['PixelArraySize']
        picam2.configure(picam2.create_video_configuration(
            main={"format": "YUV420", "size": (640, 480)},
            controls={"ScalerCrop": (0, 0, sensor_size[0], sensor_size[1])}
        ))
        picam2.start()
        time.sleep(2)  # let AWB converge
        print("[camera] opened via picamera2 (libcamera)")
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2RGB)
            with raw_lock:
                latest_raw_frame = frame
    except Exception as e:
        print(f"[camera] picamera2 failed ({e}), falling back to cv2.VideoCapture")

    # Fallback: USB webcam or legacy V4L2
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    with _cap_lock:
        _cap = cap
    print(f"[camera] opened via cv2: {cap.isOpened()}")
    while True:
        ok, frame = cap.read()
        if ok:
            with raw_lock:
                latest_raw_frame = frame
        else:
            time.sleep(0.05)

threading.Thread(target=_camera_thread, daemon=True).start()


def _sys_stats_thread():
    """Broadcast CPU temp / usage / RAM every 2 s."""
    def _temp():
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read().strip()) / 1000, 1)
        except Exception:
            return None

    while True:
        time.sleep(2)
        d = {"temp": _temp()}
        if _psutil is not None:
            mem      = _psutil.virtual_memory()
            d["cpu"] = round(_psutil.cpu_percent(), 1)
            d["ram"] = round(mem.percent, 1)
        socketio.emit("sys_stats", d)


threading.Thread(target=_sys_stats_thread, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
# DS4 controller — callbacks + drive thread
# ══════════════════════════════════════════════════════════════════════════════

def _ds4_on_cross():
    """Cross → Emergency Stop."""
    global active_keys
    with state_lock:
        active_keys.clear()
    with motor_lock:
        motor.stop()
    drive_behavior.resume()
    print("[DS4] *** EMERGENCY STOP ***")
    socketio.emit("control_state", {"manual": manual_mode, "speed": round(current_speed, 2)})
    socketio.emit("ds4_event", {"btn": "cross", "action": "emergency_stop"})


def _ds4_on_circle():
    """Circle → Toggle Recording."""
    global recording, rec_frames, rec_session
    with rec_lock:
        if not recording:
            rec_session = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            rec_frames  = 0
            recording   = True
            print(f"[DS4] record started → {rec_session}")
        else:
            recording = False
            print(f"[DS4] record stopped — {rec_frames} frames")
        sess   = rec_session
        frames = rec_frames
        is_rec = recording
    socketio.emit("record_status", {"recording": is_rec, "frames": frames, "session": sess})
    socketio.emit("ds4_event", {"btn": "circle", "action": "toggle_record"})


def _ds4_on_square():
    """Square → Toggle Sign Detection."""
    global signs_enabled
    with state_lock:
        signs_enabled = not signs_enabled
        if not signs_enabled:
            drive_behavior.resume()
    print(f"[DS4] sign detection → {'ON' if signs_enabled else 'OFF'}")
    socketio.emit("control_state", {"signs": signs_enabled})
    socketio.emit("ds4_event", {"btn": "square", "action": "toggle_signs"})


def _ds4_on_triangle():
    """Triangle → Toggle Manual / Lane Assist."""
    global manual_mode, active_keys, _curve_list
    with state_lock:
        manual_mode = not manual_mode
        active_keys.clear()
        _curve_list.clear()
    drive_behavior.resume()
    with motor_lock:
        motor.stop()
    print(f"[DS4] mode → {'Manual' if manual_mode else 'Lane Assist'}")
    socketio.emit("control_state", {"manual": manual_mode, "speed": round(current_speed, 2)})
    socketio.emit("ds4_event", {"btn": "triangle", "action": "toggle_mode"})


def _ds4_on_l1():
    """L1 → Speed -5%."""
    global current_speed
    with state_lock:
        current_speed = max(0.10, round(current_speed - 0.05, 2))
    socketio.emit("control_state", {"speed": round(current_speed, 2)})
    socketio.emit("ds4_event", {"btn": "l1", "action": "speed_down", "speed": round(current_speed, 2)})


def _ds4_on_r1():
    """R1 → Speed +5%."""
    global current_speed
    with state_lock:
        current_speed = min(1.00, round(current_speed + 0.05, 2))
    socketio.emit("control_state", {"speed": round(current_speed, 2)})
    socketio.emit("ds4_event", {"btn": "r1", "action": "speed_up", "speed": round(current_speed, 2)})


def _ds4_on_share():
    """Share → Toggle Mock Lane Mode."""
    global mock_lane_mode, _curve_list
    with state_lock:
        mock_lane_mode = not mock_lane_mode
        _curve_list.clear()
    print(f"[DS4] mock lane → {'ON' if mock_lane_mode else 'OFF'}")
    socketio.emit("control_state", {"mock_lane": mock_lane_mode})
    socketio.emit("ds4_event", {"btn": "share", "action": "toggle_mock_lane"})


def _ds4_on_options():
    """Options → Toggle Obstacle Detection."""
    global obstacles_enabled, _obstacle_stopped
    with state_lock:
        obstacles_enabled = not obstacles_enabled
        if not obstacles_enabled:
            _obstacle_stopped = False
    print(f"[DS4] obstacle detection → {'ON' if obstacles_enabled else 'OFF'}")
    socketio.emit("control_state", {"obstacles": obstacles_enabled})
    socketio.emit("ds4_event", {"btn": "options", "action": "toggle_obstacles"})


def _ds4_on_hat(dx, dy):
    """D-pad → Adjust base speed ±5%."""
    global current_speed
    if dy != 0:
        with state_lock:
            current_speed = max(0.10, min(1.00, round(current_speed + dy * 0.05, 2)))
        socketio.emit("control_state", {"speed": round(current_speed, 2)})


# ── Wire up callbacks ──────────────────────────────────────────────────────────
ds4.on_cross    = _ds4_on_cross
ds4.on_circle   = _ds4_on_circle
ds4.on_square   = _ds4_on_square
ds4.on_triangle = _ds4_on_triangle
ds4.on_l1       = _ds4_on_l1
ds4.on_r1       = _ds4_on_r1
ds4.on_share    = _ds4_on_share
ds4.on_options  = _ds4_on_options
ds4.on_hat      = _ds4_on_hat

ds4.start()
print("[DS4] controller thread started")


def _ds4_drive_thread():
    """
    At 30 Hz, reads DS4 axes and drives motors when:
      - DS4 is connected
      - ds4_enabled is True
      - manual_mode is True
      - At least one axis has non-zero input (so keyboard/joystick can still work at rest)

    Also broadcasts ds4_state at ~5 Hz so the UI updates.
    """
    interval      = 1.0 / 30
    bcast_counter = 0
    ds4_active    = False   # pentru a înregistra (0,0) o singură dată la eliberare

    while True:
        time.sleep(interval)

        with state_lock:
            is_manual = manual_mode
            enabled   = ds4_enabled

        connected = ds4.connected

        if enabled and connected and is_manual and not _parking_active():
            speed, turn = ds4.get_axes()
            # Only override motors when DS4 is actually being touched
            if abs(speed) > 0.02 or abs(turn) > 0.02:
                eff = _cap_to_speed_limit(speed * drive_behavior.multiplier)
                with motor_lock:
                    motor.move(speed=eff, turn=turn)
                pb_record_command(eff, turn)   # înregistrare playback (DS4)
                ds4_active = True
            elif ds4_active:
                # Stick-urile s-au întors la centru → înregistrează oprirea o dată
                pb_record_command(0.0, 0.0)
                ds4_active = False

        # Broadcast DS4 state at ~5 Hz
        bcast_counter += 1
        if bcast_counter >= 6:
            bcast_counter = 0
            try:
                d = ds4.get_state_dict()
                d["enabled"] = enabled
                socketio.emit("ds4_state", d)
            except Exception:
                pass


threading.Thread(target=_ds4_drive_thread, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
# Lane detection (headless — no OpenCV GUI windows)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_lane(img: np.ndarray):
    """
    Returns (annotated_frame, curve, confidence).  curve ∈ [-1, 1], confidence ∈ [0, 100].

    ROI din partea de jos + centroid stânga/dreapta:
      1. ROI = 40% din josul imaginii (imediat în față)
      2. Mască galbenă HSV
      3. Jumătatea stângă  → centroid X linie stângă
         Jumătatea dreaptă → centroid X linie dreaptă
      4. lane_center = (left_x + right_x) / 2
      5. error = lane_center - image_center
      6. Smoothing rolling average + deadband
    """
    global _curve_list, _lane_curve_smooth, _lane_raw_hist
    h_orig, w_orig = img.shape[:2]

    # ── 1. ROI: jumătatea de jos a imaginii (vede și curvele care urmează) ────
    roi_start = int(h_orig * 0.40)
    roi       = img[roi_start:, :]
    roi_proc  = cv2.resize(roi, (320, 120))
    rh, rw    = roi_proc.shape[:2]
    half      = rw // 2

    # ── 2. Mască galbenă ─────────────────────────────────────────────────────
    blurred   = cv2.GaussianBlur(roi_proc, (5, 5), 0)
    hsv       = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask      = cv2.inRange(hsv, np.array([10, 100, 80]), np.array([45, 255, 255]))
    k         = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask      = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    yellow_px = int(np.count_nonzero(mask))

    # ── 3. Centroid X per jumătate (ponderat după densitatea de galben) ───────
    # Media e PONDERATĂ după câți pixeli galbeni are fiecare coloană → un reflex
    # izolat nu mai trage centroidul. Cere un minim de galben pe fiecare parte ca
    # s-o considerăm linie validă (filtru de zgomot).
    col_l = mask[:, :half].sum(axis=0).astype(np.float32)
    col_r = mask[:, half:].sum(axis=0).astype(np.float32)
    MIN_SIDE = 255.0 * 10   # ~10 pixeli galbeni pe acea parte

    if col_l.sum() >= MIN_SIDE:
        left_x = int((np.arange(half) * col_l).sum() / col_l.sum())
    else:
        left_x = None
    if col_r.sum() >= MIN_SIDE:
        right_x = int((np.arange(half, rw) * col_r).sum() / col_r.sum())
    else:
        right_x = None

    # ── 4. Centrul benzii ────────────────────────────────────────────────────
    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) // 2
        confidence  = 100
    elif left_x is not None:
        lane_center = left_x + half // 2
        confidence  = 40
    elif right_x is not None:
        lane_center = right_x - half // 2
        confidence  = 40
    else:
        lane_center = half
        confidence  = 0

    # ── 5. Error față de centrul imaginii ────────────────────────────────────
    rawCurve = lane_center - half

    # Boost când e vizibilă doar o singură linie
    if confidence == 40:
        rawCurve = int(rawCurve * 2.0)

    if abs(rawCurve) < 3:
        rawCurve = 0

    # ── 6. Normalizare agresivă — /35 în loc de /128 ─────────────────────────
    if yellow_px > 30:
        raw_curve = max(-1.0, min(1.0, rawCurve / 35.0))
    else:
        raw_curve = 0.0

    # ── 7. Netezire temporală: median → EMA → limită de pantă (anti-spazz) ────
    # a) Median pe ultimele N cadre: respinge complet un vârf izolat de 1 cadru
    #    (cauza principală a „spasmelor" — o linie care pâlpâie un singur cadru).
    _lane_raw_hist.append(raw_curve)
    if len(_lane_raw_hist) > LANE_MEDIAN_N:
        _lane_raw_hist.pop(0)
    med = sorted(_lane_raw_hist)[len(_lane_raw_hist) // 2]
    # b) EMA: amestecă valoarea nouă cu cea anterioară → curbe line, fără salturi.
    ema  = LANE_SMOOTH_ALPHA * _lane_curve_smooth + (1.0 - LANE_SMOOTH_ALPHA) * med
    # c) Limită de pantă: curba nu se schimbă cu mai mult de LANE_MAX_STEP/cadru.
    step = max(-LANE_MAX_STEP, min(LANE_MAX_STEP, ema - _lane_curve_smooth))
    _lane_curve_smooth = max(-1.0, min(1.0, _lane_curve_smooth + step))
    curve = _lane_curve_smooth

    # ── Annotate ─────────────────────────────────────────────────────────────
    out     = img.copy()
    scale_x = w_orig / rw
    cx_img  = w_orig // 2
    cx_lane = int(lane_center * scale_x)
    y_mid   = roi_start + (h_orig - roi_start) // 2

    cv2.rectangle(out, (0, roi_start), (w_orig, h_orig), (40, 40, 40), 1)
    cv2.line(out, (cx_img, y_mid), (cx_lane, y_mid), (0, 255, 255), 2)
    cv2.circle(out, (cx_lane, y_mid), 6, (0, 255, 0), -1)
    cv2.circle(out, (cx_img,  y_mid), 4, (255, 255, 255), -1)

    offset = int(curve * 120)
    cv2.arrowedLine(out, (cx_img, h_orig - 15),
                    (cx_img + offset, int(h_orig * 0.6)),
                    (0, 255, 0), 3, tipLength=0.3)

    lbl = f"L={'ok' if left_x else '--'} R={'ok' if right_x else '--'} err:{rawCurve:+.0f} crv:{curve:+.2f}"
    cv2.putText(out, lbl, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(out, "LANE ASSIST", (w_orig - 190, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return out, curve, confidence

def _detect_lane_direct(img: np.ndarray) -> tuple:
    """
    Lane detection WITHOUT bird's eye warp — designed for small/tight tracks
    where the road goes out of frame during perspective transform.

    Algorithm:
      1. Crop to ROI (bottom portion of frame, closest to car)
      2. Threshold (same robust pipeline) to find lane pixels
      3. Sliding windows: find left & right lane bases via histogram,
         then slide N windows upward tracking the centroids
      4. Compute lane midpoint → curve ∈ [-1, 1]

    Works even when only one lane boundary is visible.
    """
    global _curve_list

    h_orig, w_orig = img.shape[:2]

    with state_lock:
        frac = roi_top_frac

    roi_y = int(h_orig * frac)
    roi   = img[roi_y:, :]

    # Process at fixed internal resolution
    proc  = cv2.resize(roi, (480, 120))
    thres = thresholding_robust(proc)
    ph, pw = thres.shape
    mid    = pw // 2

    # ── Histogram on bottom third of ROI → find starting X for each lane ──────
    hist     = np.sum(thres[ph * 2 // 3 :, :], axis=0).astype(np.float32)
    # Smooth histogram to reduce noise
    hist     = cv2.GaussianBlur(hist.reshape(1, -1), (1, 15), 0).flatten()
    left_x   = int(np.argmax(hist[:mid]))
    right_x  = int(np.argmax(hist[mid:]) + mid)

    # ── Sliding windows ────────────────────────────────────────────────────────
    N_WIN  = 7      # number of windows vertically
    MARGIN = 38     # half-width of each window (pixels at 480px resolution)
    MIN_PIX = 25    # minimum pixels to recenter window

    win_h  = ph // N_WIN
    ny, nx = np.nonzero(thres)
    lx, rx = left_x, right_x

    left_pts_x  = []
    right_pts_x = []
    win_debug   = []   # for visualization

    for w_idx in range(N_WIN):
        y_lo = ph - (w_idx + 1) * win_h
        y_hi = ph - w_idx * win_h

        l_mask = (ny >= y_lo) & (ny < y_hi) & (nx >= lx - MARGIN) & (nx < lx + MARGIN)
        r_mask = (ny >= y_lo) & (ny < y_hi) & (nx >= rx - MARGIN) & (nx < rx + MARGIN)

        gl = l_mask.nonzero()[0]
        gr = r_mask.nonzero()[0]

        win_debug.append(((lx, rx, y_lo, y_hi), len(gl), len(gr)))

        if len(gl) >= MIN_PIX:
            lx = int(np.mean(nx[gl]))
            left_pts_x.extend(nx[gl].tolist())
        if len(gr) >= MIN_PIX:
            rx = int(np.mean(nx[gr]))
            right_pts_x.extend(nx[gr].tolist())

    has_l = len(left_pts_x)  >= MIN_PIX
    has_r = len(right_pts_x) >= MIN_PIX

    if has_l and has_r:
        lane_mid = (np.mean(left_pts_x) + np.mean(right_pts_x)) / 2.0
    elif has_l:
        # Only left lane visible — estimate center is ~25% of width to the right
        lane_mid = np.mean(left_pts_x) + pw * 0.22
    elif has_r:
        lane_mid = np.mean(right_pts_x) - pw * 0.22
    else:
        lane_mid = mid   # no detection — go straight

    # ── Smooth curve via rolling average ──────────────────────────────────────
    raw = lane_mid - mid
    _curve_list.append(raw)
    if len(_curve_list) > 10:
        _curve_list.pop(0)
    # Normalize by half-width (with slight scaling for more responsive turning)
    curve = max(-1.0, min(1.0, sum(_curve_list) / len(_curve_list) / (mid * 0.75)))

    # ── Annotate frame ────────────────────────────────────────────────────────
    out     = img.copy()
    scale_x = w_orig / pw
    scale_y = (h_orig - roi_y) / ph

    # ROI boundary line
    cv2.line(out, (0, roi_y), (w_orig, roi_y), (70, 70, 70), 1)
    cv2.putText(out, f"ROI {int((1-frac)*100)}%",
                (4, roi_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)

    # Draw sliding windows (only the occupied ones)
    for (lxw, rxw, y_lo, y_hi), nl, nr in win_debug:
        oy = roi_y
        if nl >= MIN_PIX:
            x1 = int((lxw - MARGIN) * scale_x); x2 = int((lxw + MARGIN) * scale_x)
            y1 = int(y_lo * scale_y + oy);       y2 = int(y_hi * scale_y + oy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 140, 0), 1)
        if nr >= MIN_PIX:
            x1 = int((rxw - MARGIN) * scale_x); x2 = int((rxw + MARGIN) * scale_x)
            y1 = int(y_lo * scale_y + oy);       y2 = int(y_hi * scale_y + oy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 255), 1)

    # Lane centroid markers
    oy = roi_y
    if has_l:
        lx_o = int(np.mean(left_pts_x) * scale_x)
        cv2.circle(out, (lx_o, int(ph * 0.6 * scale_y + oy)), 9, (255, 140, 0), 2)
        cv2.putText(out, "L", (lx_o - 5, int(ph * 0.6 * scale_y + oy) - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 1)
    if has_r:
        rx_o = int(np.mean(right_pts_x) * scale_x)
        cv2.circle(out, (rx_o, int(ph * 0.6 * scale_y + oy)), 9, (0, 140, 255), 2)
        cv2.putText(out, "R", (rx_o - 5, int(ph * 0.6 * scale_y + oy) - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

    # Direction arrow
    lm_x = int(lane_mid * scale_x)
    cv2.arrowedLine(out, (w_orig // 2, h_orig - 20),
                    (lm_x, roi_y + 25), (0, 255, 100), 3, tipLength=0.3)

    # Confidence: window hit rate + lane quality
    window_hits  = sum(1 for (_, nl, nr) in win_debug if nl >= MIN_PIX or nr >= MIN_PIX)
    lane_quality = 1.0 if (has_l and has_r) else 0.55 if (has_l or has_r) else 0.10
    confidence   = int(min(100, window_hits / N_WIN * 55 + lane_quality * 45))

    # Info HUD
    det = "L+R" if has_l and has_r else ("L" if has_l else ("R" if has_r else "—"))
    cv2.putText(out, f"DIRECT  curve:{curve:+.2f}  [{det}]  Conf:{confidence}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
    cv2.putText(out, "SMALL TRACK",
                (w_orig - 185, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 120), 2)

    return out, curve, confidence


def _draw_warp_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw the warp trapezoid scaled to the actual frame resolution."""
    hF, wF = frame.shape[:2]
    sx = wF / 480.0
    sy = hF / 240.0

    with _warp_lock:
        wTop, hTop, wBot, hBot = WARP_POINTS

    # Four corners: TL, TR, BR, BL  (same order as warpImg)
    pts = np.array([
        [int(wTop        * sx), int(hTop * sy)],
        [int((480-wTop)  * sx), int(hTop * sy)],
        [int((480-wBot)  * sx), int(hBot * sy)],
        [int(wBot        * sx), int(hBot * sy)],
    ], dtype=np.int32)

    out = frame.copy()
    # Filled semi-transparent quad
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], (0, 180, 255))
    cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
    # Border
    cv2.polylines(out, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
    # Corner dots + labels
    labels = ["TL", "TR", "BR", "BL"]
    for (px, py), lbl in zip(pts, labels):
        cv2.circle(out, (px, py), 7, (0, 255, 0), -1)
        cv2.putText(out, lbl, (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    # Current values
    cv2.putText(out, f"WARP  wT={wTop} hT={hTop} wB={wBot} hB={hBot}",
                (6, hF - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Recording — writer thread + monitor thread
# ══════════════════════════════════════════════════════════════════════════════

def _writer_thread():
    """Drains the recording queue and writes images + CSV rows to disk."""
    global rec_frames
    pending: list = []
    last_flush    = time.time()
    csv_path      = None
    images_dir    = None
    session_name  = None

    while True:
        # Pull items off the queue
        try:
            item = _rec_queue.get(timeout=0.25)
        except queue.Empty:
            # Flush CSV if we have buffered rows
            if pending and csv_path and (time.time() - last_flush) > 1.0:
                _flush_csv(csv_path, pending)
                pending.clear()
                last_flush = time.time()
            continue

        frame, speed, turn, idx, sess = item

        # If session changed, point to the new directory
        if sess != session_name:
            if pending and csv_path:
                _flush_csv(csv_path, pending)
                pending.clear()
            session_name = sess
            images_dir   = os.path.join(DATASET_DIR, sess, "images")
            csv_path     = os.path.join(DATASET_DIR, sess, "labels.csv")
            os.makedirs(images_dir, exist_ok=True)
            # Write CSV header if file is new
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(["filename", "timestamp", "speed", "turn"])

        fname = f"{idx:06d}.jpg"
        cv2.imwrite(os.path.join(images_dir, fname), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        pending.append([fname, ts, round(speed, 4), round(turn, 4)])

        with rec_lock:
            rec_frames += 1

        # Flush every 30 frames or every second
        if len(pending) >= 30 or (time.time() - last_flush) > 1.0:
            _flush_csv(csv_path, pending)
            pending.clear()
            last_flush = time.time()


def _flush_csv(path: str, rows: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerows(rows)


threading.Thread(target=_writer_thread, daemon=True).start()


def _record_monitor_thread():
    """At RECORD_FPS, enqueues frames when recording is active."""
    global rec_frames
    interval = 1.0 / RECORD_FPS
    idx      = 0
    while True:
        time.sleep(interval)
        with rec_lock:
            if not recording:
                continue
            sess = rec_session

        frame = None
        with raw_lock:
            if latest_raw_frame is not None:
                frame = latest_raw_frame.copy()
        if frame is None:
            continue

        with state_lock:
            if manual_mode:
                spd, trn = _compute_motor_values()
            else:
                spd  = current_speed * drive_behavior.multiplier
                trn  = lane_curve

        idx += 1
        try:
            _rec_queue.put_nowait((frame, spd, trn, idx, sess))
        except queue.Full:
            pass  # drop frame rather than block


threading.Thread(target=_record_monitor_thread, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
# Sign detection thread
# ══════════════════════════════════════════════════════════════════════════════

def _sign_thread():
    global latest_detections
    interval = 1.0 / SIGN_FPS
    while True:
        time.sleep(interval)

        with state_lock:
            if not signs_enabled:
                continue

        frame = None
        with raw_lock:
            if latest_raw_frame is not None:
                frame = latest_raw_frame.copy()
        if frame is None:
            continue

        frame = _apply_cam_adjust(frame)

        try:
            dets = sign_detector.detect(frame)
        except Exception as exc:
            print(f"[signs] detection error: {exc}")
            dets = []

        mult, sign = drive_behavior.update(dets)

        with detections_lock:
            latest_detections = dets

        # ── Speed limit parsing ───────────────────────────────────────────────
        for det in dets:
            lbl = det.get("label", "")
            if lbl in SPEED_LIMIT_MAP:
                lim_frac = SPEED_LIMIT_MAP[lbl]
                with speed_limit_lock:
                    global active_speed_limit_frac
                    active_speed_limit_frac = lim_frac
                if lim_frac is None:
                    kmh_str = "∞"
                else:
                    # Folosim tabelul de display (80 → "90", etc.)
                    kmh_str = SPEED_LIMIT_DISPLAY.get(lbl) or lbl.split("_")[-1]
                socketio.emit("speed_limit_update", {
                    "label":  lbl,
                    "frac":   lim_frac,
                    "kmh":    kmh_str,
                    "active": lim_frac is not None,
                })
                print(f"[signs] ⚡ limită viteză → {kmh_str} km/h (×{lim_frac or 0:.0%})")
                break   # procesează doar primul semn de viteză per frame

        # Broadcast sign update to all browser clients
        socketio.emit("sign_update", {
            "detections": [{"label": d["label"],
                            "confidence": d["confidence"],
                            "area_frac":  d["area_frac"]} for d in dets],
            "behavior":   drive_behavior.state,
            "multiplier": round(mult, 2),
            "sign":       sign,
        })


threading.Thread(target=_sign_thread, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Obstacle detection thread
# ══════════════════════════════════════════════════════════════════════════════

def _obstacle_thread():
    global latest_obstacle, _obstacle_stopped
    interval = 1.0 / 10   # 10 Hz
    while True:
        time.sleep(interval)

        with state_lock:
            if not obstacles_enabled:
                continue
            is_auto    = not manual_mode
            signs_on   = signs_enabled

        frame = None
        with raw_lock:
            if latest_raw_frame is not None:
                frame = latest_raw_frame.copy()
        if frame is None:
            continue

        # Casetele semnelor detectate → excludem zona ca să nu fie luate ca obstacole
        sign_boxes = []
        if signs_on:
            with detections_lock:
                sign_boxes = [d["bbox"] for d in latest_detections if d.get("bbox")]

        try:
            result = obstacle_detector.detect(frame, exclude_boxes=sign_boxes)
        except Exception as exc:
            print(f"[obstacle] error: {exc}")
            continue

        with obstacle_lock:
            latest_obstacle = result

        # Auto-stop motors when dangerously close (auto mode only)
        if is_auto and result.get("dangerous"):
            if not _obstacle_stopped:
                _obstacle_stopped = True
                with motor_lock:
                    motor.stop()
                print("[obstacle] *** OBSTACLE STOP ***")
        elif _obstacle_stopped and not result.get("dangerous"):
            _obstacle_stopped = False   # obstacle cleared — lane assist resumes

        socketio.emit("obstacle_alert", {
            "detected":      result["detected"],
            "area_frac":     result["area_frac"],
            "distance_frac": result["distance_frac"],
            "dangerous":     result.get("dangerous", False),
        })


threading.Thread(target=_obstacle_thread, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Ultrasonic crash-detection thread
# ══════════════════════════════════════════════════════════════════════════════

def _ultrasonic_thread():
    global ultrasonic_cm, ultrasonic_all, _ultrasonic_stopped
    if _ultra is None:
        return
    interval = 1.0 / 15  # 15 Hz
    while True:
        time.sleep(interval)

        dists  = _ultra.get_distances()      # {name: cm|None} pentru toți 6 senzorii
        fronts = [dists.get(k) for k in ("front_left", "front_center", "front_right")]
        front_vals = [v for v in fronts if v is not None]
        front_min  = min(front_vals) if front_vals else None

        with ultrasonic_lock:
            enabled        = ultrasonic_enabled
            ultrasonic_all = dists
            ultrasonic_cm  = front_min

        # Nu interfera cu manevra de parcare — ea are propria logică de siguranță
        if _parking_active():
            socketio.emit("ultrasonic_update", {
                "cm": front_min, "dangerous": False,
                "enabled": enabled, "distances": dists,
            })
            continue

        dangerous = enabled and front_min is not None and front_min < CRASH_DISTANCE_CM

        if dangerous and not _ultrasonic_stopped:
            _ultrasonic_stopped = True
            with motor_lock:
                motor.stop()
            print(f"[ultrasonic] *** CRASH STOP — {front_min} cm (față) ***")
        elif not dangerous and _ultrasonic_stopped:
            _ultrasonic_stopped = False
            print(f"[ultrasonic] cale liberă — {front_min} cm, reiau")

        socketio.emit("ultrasonic_update", {
            "cm":        front_min,
            "dangerous": dangerous,
            "enabled":   enabled,
            "distances": dists,
        })


threading.Thread(target=_ultrasonic_thread, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Lane-assist background thread
# ══════════════════════════════════════════════════════════════════════════════

def _lane_assist_thread():
    global lane_curve, latest_annotated_frame
    interval = 1.0 / LANE_FPS
    while True:
        time.sleep(interval)
        with state_lock:
            is_auto   = not manual_mode
            spd       = current_speed
            use_bc    = behavioral_mode
            use_mock  = mock_lane_mode
            mock_curv = mock_lane_curve
            use_pid   = pid_enabled
            use_adap  = adaptive_speed_enabled
            c_k       = curve_k
            l_mode    = lane_mode

        if not is_auto:
            continue

        # Parcarea automată deține motoarele → lane assist stă deoparte
        if _parking_active():
            continue

        # Dacă playback e activ → motoarele sunt controlate de _pb_play_thread
        # dar afișăm săgeata cu virajul curent pe camera live
        with _pb_lock:
            if _pb_playing:
                frame = None
                with raw_lock:
                    if latest_raw_frame is not None:
                        frame = latest_raw_frame.copy()
                if frame is not None:
                    frame = _apply_cam_adjust(frame)
                    h, w  = frame.shape[:2]
                    trn   = _pb_current_turn
                    offset = int(trn * 150)
                    cv2.arrowedLine(frame,
                                    (w // 2, h - 20),
                                    (w // 2 + offset, int(h * 0.55)),
                                    (0, 220, 255), 3, tipLength=0.3)
                    cv2.putText(frame, f"PLAYBACK  turn:{trn:+.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                    with annotated_lock:
                        latest_annotated_frame = frame
                continue

        frame = None
        with raw_lock:
            if latest_raw_frame is not None:
                frame = latest_raw_frame.copy()
        if frame is None:
            continue

        # Aplică aceleași ajustări de cameră ca în stream-ul video afișat
        # (luminozitate / contrast / saturație setate din UI → afectează și detecția)
        frame = _apply_cam_adjust(frame)

        try:
            lane_conf = -1  # -1 = N/A (mock / behavioral); overwritten by real detectors

            if use_mock:
                # Mock lane: use curve set manually via dashboard slider
                curve     = mock_curv
                annotated = frame.copy()
                h, w = annotated.shape[:2]
                offset = int(curve * 150)
                cv2.arrowedLine(annotated, (w // 2, h - 20),
                                (w // 2 + offset, int(h * 0.55)),
                                (180, 0, 255), 3, tipLength=0.3)
                cv2.putText(annotated, f"MOCK  curve: {curve:+.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 0, 255), 2)
                cv2.putText(annotated, "DEMO MODE",
                            (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 255), 2)
            elif use_bc:
                # Behavioral cloning: CNN predicts turn directly from image
                with behavioral_lock:
                    driver = _behavioral_driver
                if driver is None:
                    continue
                curve = driver.predict(frame)
                annotated = frame.copy()
                h, w = annotated.shape[:2]
                offset = int(curve * 150)
                cv2.arrowedLine(annotated, (w // 2, h - 20),
                                (w // 2 + offset, int(h * 0.55)),
                                (255, 100, 0), 3, tipLength=0.3)
                cv2.putText(annotated, f"BC Turn: {curve:+.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 0), 2)
                cv2.putText(annotated, "BEHAVIORAL",
                            (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            elif l_mode == "direct":
                annotated, curve, lane_conf = _detect_lane_direct(frame)
            else:
                annotated, curve, lane_conf = _detect_lane(frame)

            with state_lock:
                lane_curve = curve
                lane_confidence = lane_conf
            with annotated_lock:
                latest_annotated_frame = annotated

            # ── PID turn correction ───────────────────────────────────────────
            if use_pid:
                turn = pid.compute(curve)
            else:
                turn = max(-MAX_TURN, min(MAX_TURN, curve))
            spd = current_speed

            # Limitează virajul maxim cu valoarea din slider
            turn = max(-MAX_TURN, min(MAX_TURN, turn))

            # ── Adaptive speed — slow down / stop in curves ───────────────────
            # Formula liniară: adap = 1 - c_k * |curve|
            # c_k=0.5 → la curve=1.0 merge cu 50% viteză
            # c_k=1.0 → la curve=1.0 se OPREȘTE complet
            # c_k=1.5 → se oprește deja la curve=0.67
            adap_factor = 1.0
            if use_adap:
                adap_factor = max(0.0, 1.0 - c_k * abs(curve))

            # ── Block motors if obstacle or ultrasonic crash detection triggered ─
            with obstacle_lock:
                obs_dangerous = latest_obstacle.get("dangerous", False)

            if (not obs_dangerous or not obstacles_enabled) and not _ultrasonic_stopped:
                # ── Speed Limit Enforcement — mașina merge LA viteza din semn ──
                # Semnul memorat înlocuiește viteza din slider (obey, nu doar cap),
                # apoi se aplică încetinirea în curbă și starea stop/yield.
                base_spd = spd
                with speed_limit_lock:
                    sl_frac = active_speed_limit_frac
                    sl_on   = speed_limit_enabled
                if sl_on and sl_frac is not None:
                    base_spd = sl_frac
                effective_speed = base_spd * drive_behavior.multiplier * adap_factor
                with motor_lock:
                    motor.move(speed=effective_speed, turn=turn)
            else:
                effective_speed = 0.0
                turn = 0.0
                adap_factor = 0.0

            pid_state = pid.get_state() if use_pid else None
            socketio.emit("telemetry", {
                "curve":        round(curve, 3),
                "mode":         "bc" if use_bc else ("mock" if use_mock else l_mode),
                "speed":        round(effective_speed, 2),
                "turn":         round(turn, 2),
                "adap_factor":  round(adap_factor, 3),
                "pid":          pid_state,
                "confidence":   lane_conf,
            })
        except Exception as exc:
            import traceback
            print(f"[lane] error: {exc}")
            traceback.print_exc()


threading.Thread(target=_lane_assist_thread, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
# MJPEG stream generator
# ══════════════════════════════════════════════════════════════════════════════

def _generate_frames():
    interval = 1.0 / STREAM_FPS
    while True:
        with state_lock:
            is_manual = manual_mode

        if is_manual:
            with raw_lock:
                frame = latest_raw_frame.copy() if latest_raw_frame is not None else None
        else:
            with annotated_lock:
                frame = latest_annotated_frame.copy() if latest_annotated_frame is not None else None
            if frame is None:
                with raw_lock:
                    frame = latest_raw_frame.copy() if latest_raw_frame is not None else None

        if frame is None:
            time.sleep(interval)
            continue

        # Software camera adjustments
        frame = _apply_cam_adjust(frame)

        # Warp calibration overlay (when calibration card is open)
        if _calib_overlay:
            frame = _draw_warp_overlay(frame)

        # Overlay sign bounding boxes when detection is on
        with state_lock:
            s_enabled = signs_enabled
        if s_enabled:
            with detections_lock:
                dets = list(latest_detections)
            if dets:
                frame = annotate_frame(frame, dets)

        # Overlay obstacle bounding box
        with state_lock:
            o_enabled = obstacles_enabled
        if o_enabled:
            with obstacle_lock:
                obs = dict(latest_obstacle)
            if obs.get("detected"):
                frame = obstacle_detector.draw(frame, obs)

        # Recording indicator
        with rec_lock:
            is_rec   = recording
            r_frames = rec_frames
        if is_rec:
            cv2.circle(frame, (frame.shape[1] - 22, 22), 8, (0, 0, 230), -1)
            cv2.putText(frame, f"REC {r_frames}", (frame.shape[1] - 110, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 230), 2)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes()
                   + b"\r\n")
        time.sleep(interval)

# ══════════════════════════════════════════════════════════════════════════════
# Auth helpers
# ══════════════════════════════════════════════════════════════════════════════

def _authed():
    return session.get("authenticated", False)

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not _authed():
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapped

def _valid(u, p):
    return hmac.compare_digest(u, USERNAME) and hmac.compare_digest(p, PASSWORD)

def _safe_next(nxt):
    return nxt if (nxt and nxt.startswith("/") and not nxt.startswith("//")) else url_for("index")

# ══════════════════════════════════════════════════════════════════════════════
# HTTP routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    return render_template("index.html",
                           sign_detection_available=sign_detector.available)


@app.route("/login", methods=["GET", "POST"])
def login():
    if _authed():
        return redirect(url_for("index"))
    nxt = _safe_next(request.args.get("next") or request.form.get("next"))
    if request.method == "POST":
        u        = request.form.get("username", "").strip()
        p        = request.form.get("password", "")
        remember = request.form.get("remember") == "on"
        if _valid(u, p):
            session.clear()
            session["authenticated"] = True
            session.permanent        = remember
            return redirect(nxt)
        flash("Invalid username or password.")
    return render_template("login.html", next_url=nxt)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(_generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/dataset_info")
@login_required
def dataset_info():
    """Returns a JSON summary of the dataset on disk."""
    sessions = []
    if os.path.isdir(DATASET_DIR):
        for name in sorted(os.listdir(DATASET_DIR)):
            img_dir = os.path.join(DATASET_DIR, name, "images")
            if os.path.isdir(img_dir):
                n = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
                sessions.append({"session": name, "frames": n})
    total = sum(s["frames"] for s in sessions)
    return jsonify({"sessions": sessions, "total_frames": total})


@app.route("/sessions")
@login_required
def sessions_page():
    return render_template("sessions.html")


@app.route("/sessions/list")
@login_required
def sessions_list():
    result = []
    if os.path.isdir(DATASET_DIR):
        for name in sorted(os.listdir(DATASET_DIR), reverse=True):
            img_dir = os.path.join(DATASET_DIR, name, "images")
            csv_path = os.path.join(DATASET_DIR, name, "labels.csv")
            if not os.path.isdir(img_dir):
                continue
            frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            n = len(frames)
            avg_speed, avg_turn = 0.0, 0.0
            duration_s = 0.0
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, newline="") as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        speeds = [abs(float(r["speed"])) for r in rows if r.get("speed")]
                        turns  = [abs(float(r["turn"]))  for r in rows if r.get("turn")]
                        avg_speed = round(sum(speeds) / len(speeds), 3) if speeds else 0.0
                        avg_turn  = round(sum(turns)  / len(turns),  3) if turns  else 0.0
                        ts_first = rows[0].get("timestamp", "")
                        ts_last  = rows[-1].get("timestamp", "")
                        if ts_first and ts_last:
                            try:
                                fmt = "%Y-%m-%d %H:%M:%S.%f"
                                duration_s = (
                                    datetime.strptime(ts_last,  fmt) -
                                    datetime.strptime(ts_first, fmt)
                                ).total_seconds()
                            except ValueError:
                                pass
                except Exception:
                    pass
            result.append({
                "session":    name,
                "frames":     n,
                "avg_speed":  avg_speed,
                "avg_turn":   avg_turn,
                "duration_s": round(duration_s, 1),
            })
    return jsonify(result)


@app.route("/sessions/<session_id>")
@login_required
def session_detail(session_id):
    sess_dir = os.path.join(DATASET_DIR, session_id)
    img_dir  = os.path.join(sess_dir, "images")
    csv_path = os.path.join(sess_dir, "labels.csv")
    if not os.path.isdir(img_dir):
        return jsonify({"error": "session not found"}), 404

    frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    labels = {}
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    labels[row["filename"]] = {
                        "speed": float(row.get("speed", 0)),
                        "turn":  float(row.get("turn", 0)),
                        "ts":    row.get("timestamp", ""),
                    }
        except Exception:
            pass

    frame_data = [{"idx": i, "file": fn, **labels.get(fn, {"speed": 0, "turn": 0, "ts": ""})}
                  for i, fn in enumerate(frames)]
    return jsonify({"session": session_id, "frames": frame_data})


@app.route("/sessions/<session_id>/frame/<int:idx>")
@login_required
def session_frame(session_id, idx):
    img_dir = os.path.join(DATASET_DIR, session_id, "images")
    if not os.path.isdir(img_dir):
        return ("not found", 404)
    frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if idx < 0 or idx >= len(frames):
        return ("out of range", 404)
    path = os.path.join(img_dir, frames[idx])
    return send_file(path, mimetype="image/jpeg")


@app.route("/sessions/<session_id>/delete", methods=["POST"])
@login_required
def session_delete(session_id):
    sess_dir = os.path.join(DATASET_DIR, session_id)
    if os.path.isdir(sess_dir) and session_id.startswith("session_"):
        shutil.rmtree(sess_dir)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "not found"}), 404


# ── Training routes (Behavioral Cloning) ──────────────────────────

_training_lock   = threading.Lock()
_training_active = False
_training_status = {}   # last progress dict


@app.route("/train")
@login_required
def train_page():
    sessions = []
    if os.path.isdir(DATASET_DIR):
        for name in sorted(os.listdir(DATASET_DIR)):
            img_dir = os.path.join(DATASET_DIR, name, "images")
            if os.path.isdir(img_dir):
                n = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
                sessions.append({"session": name, "frames": n})
    models_dir = os.path.join(_BASE, "models")
    models = sorted(os.listdir(models_dir)) if os.path.isdir(models_dir) else []
    return render_template("training.html",
                           sessions=sessions,
                           models=models,
                           training_active=_training_active,
                           training_status=_training_status)


@app.route("/train/start", methods=["POST"])
@login_required
def train_start():
    global _training_active
    with _training_lock:
        if _training_active:
            return jsonify({"ok": False, "error": "Training deja în curs"}), 400

    data     = request.get_json(force=True)
    sess_ids = data.get("sessions", [])
    epochs   = int(data.get("epochs", 20))
    lr       = float(data.get("lr", 1e-3))

    # Validate sessions exist
    valid = []
    for s in sess_ids:
        p = os.path.join(DATASET_DIR, s)
        if os.path.isdir(p) and s.startswith("session_"):
            valid.append(p)
    if not valid:
        return jsonify({"ok": False, "error": "Nicio sesiune validă selectată"}), 400

    def _run():
        global _training_active, _training_status
        with _training_lock:
            _training_active = True
        try:
            from behavioral_cloning import train_model
            def _progress(ep, total_ep, loss, val_loss):
                pct = round(ep / total_ep * 100)
                d = {"epoch": ep, "total": total_ep, "loss": round(loss, 4),
                     "val_loss": round(val_loss, 4) if val_loss is not None else None,
                     "percent": pct}
                global _training_status
                _training_status = d
                socketio.emit("training_progress", d)
            out_path = train_model(valid, epochs=epochs, lr=lr, progress_cb=_progress)
            socketio.emit("training_progress", {"done": True, "model": out_path, "percent": 100})
        except Exception as exc:
            socketio.emit("training_progress", {"error": str(exc), "percent": -1})
        finally:
            with _training_lock:
                _training_active = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/train/status")
@login_required
def train_status():
    return jsonify({"active": _training_active, **_training_status})


@app.route("/models")
@login_required
def models_list():
    models_dir = os.path.join(_BASE, "models")
    if not os.path.isdir(models_dir):
        return jsonify([])
    files = [f for f in sorted(os.listdir(models_dir)) if f.endswith(".onnx")]
    return jsonify(files)


@app.route("/warp_preview")
@login_required
def warp_preview():
    """
    Returns a JPEG with the warp trapezoid and the bird's-eye result
    side-by-side so the user can calibrate warp points via the browser.
    """
    frame = None
    with raw_lock:
        if latest_raw_frame is not None:
            frame = latest_raw_frame.copy()
    if frame is None:
        # Return a black placeholder if no camera frame yet
        frame = np.zeros((240, 480, 3), dtype=np.uint8)

    proc = cv2.resize(frame, (480, 240))
    hT, wT = proc.shape[:2]

    with _warp_lock:
        wTop, hTop, wBot, hBot = WARP_POINTS

    pts = np.float32([
        (wTop,      hTop),
        (wT - wTop, hTop),
        (wBot,      hBot),
        (wT - wBot, hBot),
    ])

    # --- Left panel: raw frame with trapezoid overlay ---
    left = proc.copy()
    poly = pts.astype(np.int32).reshape((-1, 1, 2))
    # Draw trapezoid edges
    cv2.polylines(left, [np.array([
        pts[0].astype(int), pts[1].astype(int),
        pts[3].astype(int), pts[2].astype(int),
    ])], isClosed=True, color=(0, 200, 255), thickness=2)
    # Draw corner circles
    for (px, py) in pts.astype(int):
        cv2.circle(left, (px, py), 6, (0, 255, 0), -1)
    cv2.putText(left, "RAW + WARP REGION", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # --- Right panel: mască galbenă + linii Hough detectate ---
    warped_c = utlis.warpImg(proc, pts, wT, hT)
    _sm = int(wT * 0.06); _tm = int(hT * 0.08)
    warped_c[:_tm, :] = 0; warped_c[:, :_sm] = 0; warped_c[:, wT - _sm:] = 0
    _bl   = cv2.GaussianBlur(warped_c, (5, 5), 0)
    _hsv  = cv2.cvtColor(_bl, cv2.COLOR_BGR2HSV)
    _mask = cv2.inRange(_hsv, np.array([10, 100, 80]), np.array([45, 255, 255]))
    _k    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    _mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE, _k)
    _edges = cv2.Canny(_mask, 50, 100)
    _lines = cv2.HoughLinesP(_edges, 1, np.pi / 180, threshold=10,
                              minLineLength=5, maxLineGap=150)
    right  = cv2.cvtColor(_mask, cv2.COLOR_GRAY2BGR)
    # Desenează liniile detectate
    _half = wT // 2
    _ln, _rn = [], []
    if _lines is not None:
        for _seg in _lines:
            _x1, _y1, _x2, _y2 = _seg[0]
            _cx = (_x1 + _x2) // 2
            _col = (255, 200, 0) if _cx < _half else (0, 200, 255)
            cv2.line(right, (_x1, _y1), (_x2, _y2), _col, 2)
            if _cx < _half: _ln.append(_cx)
            else:           _rn.append(_cx)
    cv2.line(right, (_half, 0), (_half, hT), (80, 80, 80), 1)
    _lpx = int(np.mean(_ln)) if _ln else None
    _rpx = int(np.mean(_rn)) if _rn else None
    _nc  = ((_lpx or _half//2) + (_rpx or _half+_half//2)) // 2
    cv2.line(right, (_nc, 0), (_nc, hT), (0, 255, 0), 2)
    _info = f"L={'ok' if _lpx else '--'} R={'ok' if _rpx else '--'} center={_nc}"
    cv2.putText(right, _info, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 120), 1)

    combined = np.hstack([left, right])   # 960×240
    ok, buf  = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ("preview failed", 500)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/snapshot")
@login_required
def snapshot():
    """Single JPEG frame — used by mobile browsers that can't play MJPEG."""
    # Serve annotated frame if in auto mode, else raw
    with state_lock:
        is_manual = manual_mode
    if not is_manual:
        with annotated_lock:
            frame = latest_annotated_frame.copy() if latest_annotated_frame is not None else None
    else:
        frame = None
    if frame is None:
        with raw_lock:
            frame = latest_raw_frame.copy() if latest_raw_frame is not None else None
    if frame is None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Apply software camera adjustments
    frame = _apply_cam_adjust(frame)

    # Overlay sign boxes on snapshot too
    with state_lock:
        s_enabled = signs_enabled
    if s_enabled:
        with detections_lock:
            dets = list(latest_detections)
        if dets:
            frame = annotate_frame(frame, dets)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ("encode failed", 500)
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp

# ══════════════════════════════════════════════════════════════════════════════
# SocketIO helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_direction():
    fwd = "w" in active_keys; bwd = "s" in active_keys
    lft = "a" in active_keys; rgt = "d" in active_keys
    if fwd and lft: return "FWD-LEFT"
    if fwd and rgt: return "FWD-RIGHT"
    if bwd and lft: return "BWD-LEFT"
    if bwd and rgt: return "BWD-RIGHT"
    if fwd: return "FORWARD"
    if bwd: return "BACKWARD"
    if lft: return "LEFT"
    if rgt: return "RIGHT"
    return "STOP"

def _compute_motor_values():
    spd = 0.0; trn = 0.0
    if "w" in active_keys: spd += current_speed
    if "s" in active_keys: spd -= current_speed
    if "a" in active_keys: trn -= BASE_TURN
    if "d" in active_keys: trn += BASE_TURN
    return max(-1.0, min(1.0, spd)), max(-1.0, min(1.0, trn))

def _emit_state():
    with _warp_lock:
        wp = list(WARP_POINTS)
    emit("control_state", {
        "manual":      manual_mode,
        "speed":       round(current_speed, 2),
        "signs":       signs_enabled,
        "recording":   recording,
        "rec_frames":  rec_frames,
        "rec_session": rec_session,
        "warp_points": wp,
    })
    # PID + adaptive speed + lane mode state
    emit("pid_state", {**pid.get_state(), "enabled": pid_enabled})
    emit("control_state", {
        "adaptive_speed":       adaptive_speed_enabled,
        "curve_k":              round(curve_k, 2),
        "lane_mode":            lane_mode,
        "roi_top_frac":         round(roi_top_frac, 2),
        "speed_limit_enabled":  speed_limit_enabled,
    })
    # Push current speed limit state
    with speed_limit_lock:
        sl_frac = active_speed_limit_frac
    emit("speed_limit_update", {"active": sl_frac is not None, "frac": sl_frac,
                                 "kmh": None})
    # Ultrasonic crash detection state + parking availability
    with ultrasonic_lock:
        dists = dict(ultrasonic_all)
        emit("control_state", {
            "ultrasonic":         ultrasonic_enabled,
            "crash_distance_cm":  CRASH_DISTANCE_CM,
            "parking_available":  parking is not None,
        })
    if dists:
        emit("ultrasonic_update", {"cm": ultrasonic_cm, "dangerous": False,
                                   "enabled": ultrasonic_enabled, "distances": dists})
    # Current parking maneuver state
    if parking is not None:
        emit("parking_status", parking.get_state())
    # Also push current DS4 state to the new client
    d = ds4.get_state_dict()
    d["enabled"] = ds4_enabled
    emit("ds4_state", d)

# ══════════════════════════════════════════════════════════════════════════════
# SocketIO events
# ══════════════════════════════════════════════════════════════════════════════

@socketio.on("connect")
def on_connect():
    if not _authed():
        return False
    _emit_state()


@socketio.on("disconnect")
def on_disconnect():
    with motor_lock:
        motor.stop()
    print("[server] client disconnected — motors stopped")


@socketio.on("drive_command")
def on_drive(data):
    global active_keys
    if not _authed():
        return
    if _parking_active():
        return                     # parking owns the motors
    with state_lock:
        if not manual_mode:
            return
    key    = str(data.get("key",    "")).lower()
    action = str(data.get("action", ""))
    if key not in ("w", "a", "s", "d"):
        return

    with state_lock:
        if action == "press":
            active_keys.add(key)
        elif action == "release":
            active_keys.discard(key)
        spd, trn = _compute_motor_values()
        direction = _get_direction()

    # Apply sign-detection multiplier + speed limit even in manual mode
    eff_spd = _cap_to_speed_limit(spd * drive_behavior.multiplier)
    with motor_lock:
        motor.move(speed=eff_spd, turn=trn)
    # Înregistrează comanda dacă recording e activ
    pb_record_command(eff_spd, trn)

    emit("telemetry", {
        "direction": direction,
        "speed":     round(eff_spd, 2),
        "turn":      round(trn, 2),
    }, broadcast=True)


@socketio.on("set_speed")
def on_set_speed(data):
    global current_speed
    v = float(data.get("speed", BASE_SPEED))
    with state_lock:
        current_speed = max(0.0, min(1.0, v))
    emit("control_state", {"manual": manual_mode, "speed": round(current_speed, 2)},
         broadcast=True)


@socketio.on("set_max_turn")
def on_set_max_turn(data):
    global MAX_TURN
    v = float(data.get("max_turn", 0.80))
    MAX_TURN = max(0.1, min(1.0, v))
    emit("control_state", {"max_turn": round(MAX_TURN, 2)}, broadcast=True)
    print(f"[server] max_turn → {MAX_TURN:.2f}")


@socketio.on("toggle_mode")
def on_toggle_mode(data):
    global manual_mode, active_keys, _curve_list, latest_annotated_frame, _lane_curve_smooth
    going_manual = bool(data.get("manual", True))
    with state_lock:
        manual_mode = going_manual
        active_keys.clear()
        _curve_list.clear()
        _lane_curve_smooth = 0.0   # pornește netezirea benzii de la zero
        _lane_raw_hist.clear()
    drive_behavior.resume()
    pid.reset()                    # curăță și integratorul/derivata PID
    with motor_lock:
        motor.stop()
    if going_manual:
        # Trecem pe Manual → oprire playback dacă era activ
        pb_stop_play()
    else:
        # Trecem pe Lane Assist → curăță frame-ul adnotat vechi
        with annotated_lock:
            latest_annotated_frame = None
    emit("control_state", {"manual": manual_mode, "speed": round(current_speed, 2)},
         broadcast=True)
    print(f"[server] mode → {'Manual' if manual_mode else 'Lane Assist'}")


@socketio.on("pb_record_start")
def on_pb_record_start():
    pb_start_record()

@socketio.on("pb_record_stop")
def on_pb_record_stop():
    pb_stop_record()

@socketio.on("pb_play_start")
def on_pb_play_start():
    pb_start_play()

@socketio.on("pb_play_stop")
def on_pb_play_stop():
    pb_stop_play()

@socketio.on("pb_clear")
def on_pb_clear():
    global _pb_sequence
    pb_stop_play()
    with _pb_lock:
        _pb_sequence = []
    socketio.emit("pb_state", {"recording": False, "playing": False, "has_sequence": False})


@socketio.on("toggle_record")
def on_toggle_record(data):
    global recording, rec_frames, rec_session
    want = bool(data.get("recording", False))
    with rec_lock:
        if want and not recording:
            rec_session = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            rec_frames  = 0
            recording   = True
            print(f"[record] started → dataset/{rec_session}/")
        elif not want and recording:
            recording = False
            print(f"[record] stopped — {rec_frames} frames saved")
        sess    = rec_session
        frames  = rec_frames
        is_rec  = recording
    emit("record_status", {
        "recording": is_rec,
        "frames":    frames,
        "session":   sess,
    }, broadcast=True)


@socketio.on("set_warp")
def on_set_warp(data):
    """Update warp points live from the browser calibration panel."""
    global WARP_POINTS, _curve_list
    try:
        pts = [int(data["wTop"]), int(data["hTop"]),
               int(data["wBot"]), int(data["hBot"])]
        # Clamp to valid range for 480×240 processing resolution
        pts[0] = max(0, min(240, pts[0]))   # wTop
        pts[1] = max(0, min(240, pts[1]))   # hTop
        pts[2] = max(0, min(240, pts[2]))   # wBot
        pts[3] = max(0, min(240, pts[3]))   # hBot
        with _warp_lock:
            WARP_POINTS = pts
        with state_lock:
            _curve_list.clear()
        print(f"[lane] warp points updated: {pts}")
        emit("warp_ack", {"warp_points": pts, "ok": True}, broadcast=True)
    except (KeyError, ValueError) as exc:
        emit("warp_ack", {"ok": False, "error": str(exc)})


@socketio.on("set_calib_overlay")
def on_set_calib_overlay(data):
    global _calib_overlay
    _calib_overlay = bool(data.get("enabled", False))


@socketio.on("toggle_signs")
def on_toggle_signs(data):
    global signs_enabled
    with state_lock:
        signs_enabled = bool(data.get("enabled", False))
        if not signs_enabled:
            drive_behavior.resume()
    print(f"[server] sign detection → {'ON' if signs_enabled else 'OFF'}")
    emit("control_state", {"signs": signs_enabled}, broadcast=True)


@socketio.on("set_mock_curve")
def on_set_mock_curve(data):
    global mock_lane_curve
    with state_lock:
        mock_lane_curve = max(-1.0, min(1.0, float(data.get("curve", 0.0))))


@socketio.on("toggle_mock_lane")
def on_toggle_mock_lane(data):
    global mock_lane_mode, _curve_list
    with state_lock:
        mock_lane_mode = bool(data.get("enabled", False))
        _curve_list.clear()
    print(f"[server] mock lane → {'ON' if mock_lane_mode else 'OFF'}")
    emit("control_state", {"mock_lane": mock_lane_mode}, broadcast=True)


@socketio.on("toggle_behavioral")
def on_toggle_behavioral(data):
    global behavioral_mode, _behavioral_driver
    model_name = data.get("model", "")
    enabled    = bool(data.get("enabled", False))

    if enabled and model_name:
        model_path = os.path.join(_BASE, "models", model_name)
        if not os.path.exists(model_path):
            emit("behavioral_status", {"ok": False, "error": "Model negăsit"})
            return
        try:
            from behavioral_cloning import BehavioralDriver
            with behavioral_lock:
                _behavioral_driver = BehavioralDriver(model_path)
            with state_lock:
                behavioral_mode = True
            print(f"[server] behavioral mode ON — {model_name}")
            emit("behavioral_status", {"ok": True, "model": model_name, "active": True},
                 broadcast=True)
        except Exception as exc:
            emit("behavioral_status", {"ok": False, "error": str(exc)})
    else:
        with state_lock:
            behavioral_mode = False
        print("[server] behavioral mode OFF")
        emit("behavioral_status", {"ok": True, "active": False}, broadcast=True)


@socketio.on("toggle_obstacles")
def on_toggle_obstacles(data):
    global obstacles_enabled, _obstacle_stopped
    with state_lock:
        obstacles_enabled = bool(data.get("enabled", False))
        if not obstacles_enabled:
            _obstacle_stopped = False
            obstacle_detector.reset()
    print(f"[server] obstacle detection → {'ON' if obstacles_enabled else 'OFF'}")
    emit("control_state", {"obstacles": obstacles_enabled}, broadcast=True)


@socketio.on("toggle_ultrasonic")
def on_toggle_ultrasonic(data):
    global ultrasonic_enabled, _ultrasonic_stopped
    with ultrasonic_lock:
        ultrasonic_enabled = bool(data.get("enabled", False))
        if not ultrasonic_enabled:
            _ultrasonic_stopped = False
    print(f"[server] ultrasonic crash detection → {'ON' if ultrasonic_enabled else 'OFF'}")
    emit("control_state", {"ultrasonic": ultrasonic_enabled}, broadcast=True)


@socketio.on("start_parking")
def on_start_parking(data=None):
    """Pornește manevra de parcare automată cu spatele."""
    global manual_mode
    if not _authed():
        return
    if parking is None:
        emit("parking_status", {"active": False, "state": "error",
                                "message": "Senzorii ultrasonici nu sunt disponibili",
                                "distances": {}}, broadcast=True)
        return
    # Parcarea conduce singură. Forțăm modul manual ca, după terminare, lane assist
    # să NU repornească brusc și să intre în peretele garajului.
    with state_lock:
        manual_mode = True
    with motor_lock:
        motor.stop()
    turn_dir = +1
    if isinstance(data, dict) and data.get("dir") in ("left", "right"):
        turn_dir = -1 if data["dir"] == "left" else +1
    ok = parking.start(turn_dir=turn_dir)
    emit("control_state", {"manual": True}, broadcast=True)
    print(f"[server] start_parking → {'OK' if ok else 'deja activ'}")


@socketio.on("stop_parking")
def on_stop_parking(data=None):
    """Anulează manevra de parcare și oprește motoarele."""
    if not _authed():
        return
    if parking is not None:
        parking.stop()
    with motor_lock:
        motor.stop()
    print("[server] stop_parking")


@socketio.on("emergency_stop")
def on_emergency_stop():
    global active_keys
    if parking is not None and parking.active:
        parking.stop()            # cancel any in-progress parking maneuver
    with state_lock:
        active_keys.clear()
    with motor_lock:
        motor.stop()
    drive_behavior.resume()
    print("[server] *** EMERGENCY STOP ***")
    emit("control_state", {"manual": manual_mode,
                           "speed":  round(current_speed, 2)}, broadcast=True)


@socketio.on("motor_move")
def on_motor_move(data):
    """Continuous speed+turn from gamepad or touch joystick."""
    if not _authed():
        return
    if _parking_active():
        return                     # parking owns the motors
    with state_lock:
        if not manual_mode:
            return
    spd = max(-1.0, min(1.0, float(data.get("speed", 0))))
    trn = max(-1.0, min(1.0, float(data.get("turn",  0))))
    eff = _cap_to_speed_limit(spd * drive_behavior.multiplier)
    with motor_lock:
        motor.move(speed=eff, turn=trn)
    pb_record_command(eff, trn)   # înregistrare playback (joystick)


@socketio.on("set_camera")
def on_set_camera(data):
    """Update software camera adjustments (no cap.set — safe for all cameras)."""
    global _cam_brightness, _cam_contrast, _cam_saturation
    with _cam_lock:
        if "brightness" in data:
            # slider 0-100 → offset -100…+100  (50 = no change)
            _cam_brightness = (float(data["brightness"]) - 50) * 2
        if "contrast" in data:
            # slider 0-100 → multiplier 0.0…2.0  (50 → 1.0 = no change)
            _cam_contrast = float(data["contrast"]) / 50.0
        if "saturation" in data:
            _cam_saturation = float(data["saturation"]) / 50.0


# ── PID socket events ─────────────────────────────────────────────────────────

@socketio.on("toggle_pid")
def on_toggle_pid(data):
    global pid_enabled
    with state_lock:
        pid_enabled = bool(data.get("enabled", False))
    if pid_enabled:
        pid.reset()   # clean state when switching on
    print(f"[server] PID → {'ON' if pid_enabled else 'OFF'}")
    emit("pid_state", {**pid.get_state(), "enabled": pid_enabled}, broadcast=True)


@socketio.on("set_pid")
def on_set_pid(data):
    kp = data.get("kp")
    ki = data.get("ki")
    kd = data.get("kd")
    pid.set_gains(
        kp=float(kp) if kp is not None else None,
        ki=float(ki) if ki is not None else None,
        kd=float(kd) if kd is not None else None,
    )
    emit("pid_state", {**pid.get_state(), "enabled": pid_enabled}, broadcast=True)


@socketio.on("reset_pid")
def on_reset_pid():
    pid.reset()
    emit("pid_state", {**pid.get_state(), "enabled": pid_enabled}, broadcast=True)


# ── Lane mode socket events ───────────────────────────────────────────────────

@socketio.on("set_lane_mode")
def on_set_lane_mode(data):
    global lane_mode, _curve_list
    mode = data.get("mode", "warp")
    if mode in ("warp", "direct"):
        with state_lock:
            lane_mode = mode
            _curve_list.clear()
        pid.reset()
        print(f"[server] lane mode → {lane_mode}")
        emit("control_state", {"lane_mode": lane_mode,
                                "roi_top_frac": round(roi_top_frac, 2)}, broadcast=True)


@socketio.on("set_roi_frac")
def on_set_roi_frac(data):
    global roi_top_frac
    with state_lock:
        roi_top_frac = max(0.20, min(0.80, float(data.get("frac", 0.55))))
    emit("control_state", {"roi_top_frac": round(roi_top_frac, 2)}, broadcast=True)


# ── Adaptive Speed socket events ───────────────────────────────────────────────

@socketio.on("toggle_adaptive_speed")
def on_toggle_adaptive_speed(data):
    global adaptive_speed_enabled
    with state_lock:
        adaptive_speed_enabled = bool(data.get("enabled", False))
    print(f"[server] adaptive speed → {'ON' if adaptive_speed_enabled else 'OFF'}")
    emit("control_state", {"adaptive_speed": adaptive_speed_enabled,
                           "curve_k": curve_k}, broadcast=True)


@socketio.on("set_curve_k")
def on_set_curve_k(data):
    global curve_k
    with state_lock:
        curve_k = max(0.0, min(1.5, float(data.get("curve_k", 0.6))))
    emit("control_state", {"curve_k": round(curve_k, 2)}, broadcast=True)


# ── Analytics routes ───────────────────────────────────────────────────────────

@app.route("/analytics")
@login_required
def analytics_page():
    return render_template("analytics.html")


@app.route("/analytics/data")
@login_required
def analytics_data():
    """Compute and return full dataset statistics as JSON."""
    import random as _random

    sessions_data = []
    all_turns     = []
    all_speeds    = []

    if os.path.isdir(DATASET_DIR):
        for name in sorted(os.listdir(DATASET_DIR)):
            img_dir  = os.path.join(DATASET_DIR, name, "images")
            csv_path = os.path.join(DATASET_DIR, name, "labels.csv")
            if not os.path.isdir(img_dir) or not os.path.exists(csv_path):
                continue

            turns  = []
            speeds = []
            try:
                with open(csv_path, newline="") as f:
                    for row in csv.DictReader(f):
                        try:
                            turns.append(float(row.get("turn",  0)))
                            speeds.append(float(row.get("speed", 0)))
                        except (ValueError, TypeError):
                            pass
            except Exception:
                continue

            if not turns:
                continue

            all_turns.extend(turns)
            all_speeds.extend(speeds)

            n          = len(turns)
            avg_turn   = sum(turns)  / n
            avg_speed  = sum(abs(s) for s in speeds) / n
            left_n     = sum(1 for t in turns if t < -0.1)
            right_n    = sum(1 for t in turns if t >  0.1)
            straight_n = n - left_n - right_n
            balance    = round((1.0 - abs(avg_turn)) * 100, 1)
            frames     = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            duration_s = frames / RECORD_FPS

            sessions_data.append({
                "session":      name,
                "frames":       frames,
                "duration_s":   round(duration_s, 1),
                "avg_turn":     round(avg_turn,  4),
                "avg_speed":    round(avg_speed, 4),
                "balance":      balance,
                "left_pct":     round(left_n     / n * 100, 1),
                "right_pct":    round(right_n    / n * 100, 1),
                "straight_pct": round(straight_n / n * 100, 1),
            })

    # ── Global turn histogram (20 bins, -1 … +1) ──────────────────────────────
    turn_bins   = [round(-1.0 + i * 0.1, 1) for i in range(20)]
    turn_counts = [0] * 20
    for t in all_turns:
        idx = min(19, max(0, int((t + 1.0) / 2.0 * 20)))
        turn_counts[idx] += 1

    # ── Global speed histogram (10 bins, 0 … 1) ───────────────────────────────
    speed_bins   = [round(i * 0.1, 1) for i in range(10)]
    speed_counts = [0] * 10
    for s in all_speeds:
        idx = min(9, max(0, int(abs(s) * 10)))
        speed_counts[idx] += 1

    # ── Scatter sample (max 600 points) ───────────────────────────────────────
    scatter_data = []
    if all_turns:
        pairs = list(zip(all_speeds, all_turns))
        if len(pairs) > 600:
            pairs = _random.sample(pairs, 600)
        scatter_data = [{"x": round(abs(s), 3), "y": round(t, 3)} for s, t in pairs]

    total_frames  = sum(s["frames"]     for s in sessions_data)
    total_hours   = round(total_frames / RECORD_FPS / 3600, 4) if total_frames else 0
    global_avg_t  = sum(all_turns) / len(all_turns) if all_turns else 0
    global_bal    = round((1.0 - abs(global_avg_t)) * 100, 1) if all_turns else 0

    return jsonify({
        "sessions":        sessions_data,
        "total_frames":    total_frames,
        "total_sessions":  len(sessions_data),
        "total_hours":     total_hours,
        "global_balance":  global_bal,
        "global_avg_turn": round(global_avg_t, 4),
        "turn_histogram":  {"bins": turn_bins,  "counts": turn_counts},
        "speed_histogram": {"bins": speed_bins, "counts": speed_counts},
        "scatter":         scatter_data,
    })


@socketio.on("toggle_ds4")
def on_toggle_ds4(data):
    """Enable / disable DS4 input from the UI."""
    global ds4_enabled
    ds4_enabled = bool(data.get("enabled", True))
    print(f"[server] DS4 input → {'ON' if ds4_enabled else 'OFF'}")
    d = ds4.get_state_dict()
    d["enabled"] = ds4_enabled
    emit("ds4_state", d, broadcast=True)
    socketio.emit("ds4_event", {"action": "toggle", "enabled": ds4_enabled})


# ── Speed Limit socket events ──────────────────────────────────────────────────

@socketio.on("toggle_speed_limit")
def on_toggle_speed_limit(data):
    global speed_limit_enabled
    with speed_limit_lock:
        speed_limit_enabled = bool(data.get("enabled", True))
    print(f"[server] speed limit enforcement → {'ON' if speed_limit_enabled else 'OFF'}")
    emit("control_state", {"speed_limit_enabled": speed_limit_enabled}, broadcast=True)


@socketio.on("clear_speed_limit")
def on_clear_speed_limit():
    global active_speed_limit_frac
    with speed_limit_lock:
        active_speed_limit_frac = None
    emit("speed_limit_update", {"active": False, "frac": None, "kmh": None}, broadcast=True)


# ── Race Timer socket events ───────────────────────────────────────────────────

@socketio.on("race_start")
def on_race_start():
    global race_active, race_start_time, race_laps, race_lap_start
    with race_lock:
        race_active     = True
        race_start_time = time.time()
        race_lap_start  = race_start_time
        race_laps       = []
    emit("race_state", _get_race_state(), broadcast=True)
    print("[race] started")


@socketio.on("lap_trigger")
def on_lap_trigger():
    global race_lap_start
    with race_lock:
        if not race_active:
            return
        now      = time.time()
        lap_time = round(now - race_lap_start, 3)
        race_laps.append(lap_time)
        race_lap_start = now
    emit("race_state", _get_race_state(), broadcast=True)
    print(f"[race] lap {len(race_laps)} → {lap_time:.3f}s")


@socketio.on("race_stop")
def on_race_stop():
    global race_active
    with race_lock:
        race_active = False
    emit("race_state", _get_race_state(), broadcast=True)
    print("[race] stopped")


@socketio.on("race_reset")
def on_race_reset():
    global race_active, race_start_time, race_laps, race_lap_start
    with race_lock:
        race_active     = False
        race_start_time = None
        race_laps       = []
        race_lap_start  = None
    emit("race_state", _get_race_state(), broadcast=True)
    print("[race] reset")


# ── Race Timer page route ──────────────────────────────────────────────────────

@app.route("/race")
@login_required
def race_page():
    return render_template("race.html")


# ── Config Manager routes ──────────────────────────────────────────────────────

@app.route("/config")
@login_required
def config_page_route():
    cfg = _load_config_file()
    return render_template("config.html",
                           config=cfg,
                           presets=list(CONFIG_PRESETS.keys()),
                           default_config=DEFAULT_CONFIG)


@app.route("/config/load")
@login_required
def config_load_route():
    return jsonify(_load_config_file())


@app.route("/config/save", methods=["POST"])
@login_required
def config_save_route():
    data   = request.get_json(force=True)
    preset = data.get("preset")
    if preset and preset in CONFIG_PRESETS:
        cfg = CONFIG_PRESETS[preset].copy()
    else:
        cfg = DEFAULT_CONFIG.copy()
        for k, default_val in DEFAULT_CONFIG.items():
            if k not in data:
                continue
            try:
                if isinstance(default_val, list):
                    cfg[k] = [int(x) for x in data[k]]
                elif isinstance(default_val, bool):
                    cfg[k] = bool(data[k])
                elif isinstance(default_val, int):
                    cfg[k] = int(data[k])
                elif isinstance(default_val, str):
                    cfg[k] = str(data[k])
                else:
                    cfg[k] = float(data[k])
            except (ValueError, TypeError):
                pass
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        _apply_config(cfg)
        return jsonify({"ok": True, "config": cfg})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ── Video Export routes ────────────────────────────────────────────────────────

@app.route("/sessions/<session_id>/export", methods=["POST"])
@login_required
def session_export_start(session_id):
    sess_dir = os.path.join(DATASET_DIR, session_id)
    img_dir  = os.path.join(sess_dir, "images")
    if not os.path.isdir(img_dir):
        return jsonify({"ok": False, "error": "Sesiune negăsită"}), 404
    with _export_lock2:
        job = _export_jobs.get(session_id)
        if job and job.get("status") == "running":
            return jsonify({"ok": False, "error": "Export deja în curs"}), 400
        _export_jobs[session_id] = {"status": "running", "progress": 0, "path": None, "error": None}
    threading.Thread(target=_export_session_thread, args=(session_id,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/sessions/<session_id>/export/status")
@login_required
def session_export_status(session_id):
    with _export_lock2:
        job = dict(_export_jobs.get(session_id, {"status": "idle"}))
    job.pop("path", None)    # don't expose internal path
    return jsonify(job)


@app.route("/sessions/<session_id>/export/download")
@login_required
def session_export_download(session_id):
    with _export_lock2:
        job = dict(_export_jobs.get(session_id, {}))
    path = job.get("path")
    if not path or not os.path.exists(path):
        return ("Export negăsit — rulează mai întâi exportul", 404)
    return send_file(path, mimetype="video/mp4",
                     as_attachment=True,
                     download_name=f"{session_id}.mp4")

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Încarcă și aplică setările salvate la pornire
    _apply_config(_load_config_file())
    print(f"[server] starting on http://0.0.0.0:{PORT}")
    print(f"[server] dataset will be saved to: {DATASET_DIR}")
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
