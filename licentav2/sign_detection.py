"""
sign_detection.py
=================
Two-stage traffic sign detection pipeline:

  Stage 1 — Color segmentation (OpenCV, ~2 ms/frame)
            Finds candidate regions that are red, blue, or yellow — the
            dominant colours of traffic signs.

  Stage 2 — ONNX classifier (OpenCV DNN, ~8–25 ms/region on Pi 4)
            Our MobileNetV2 trained on GTSRB classifies each candidate.

No GPU required.  No ultralytics dependency.
Just:  pip install opencv-python numpy

Place traffic_sign_classifier.onnx in the same directory as this file,
or set ONNX_MODEL_PATH to its full path.
"""

import os
import threading
import time
from collections import defaultdict

import cv2
import numpy as np

# ── Model path ─────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(__file__)
ONNX_MODEL_PATH = os.path.join(_HERE, "traffic_sign_classifier.onnx")
IMG_SIZE        = 96      # must match config.IMG_SIZE used during training

# ── Detection thresholds ───────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.60    # minimum classifier confidence (false positives reduse
                          # după eliminarea măștilor alb/galben + crop top 60%)
TRIGGER_AREA    = 0.008   # bbox ≥ 0.8% din cadru → proxy distanță
MIN_REGION_PX   = 900     # ignoră blobs mai mici de 900px (~30×30 px)
CONFIRM_FRAMES  = 3       # detectări consecutive necesare (anti-false-positive)
RED_FILL_STOP   = 0.30    # dacă un semn rotund/octogonal e umplut cu roșu peste
                          # acest prag → e STOP (nu limită de viteză, care are
                          # centru alb). Crește dacă semnele de viteză declanșează
                          # STOP; scade dacă STOP-ul nu e prins.

# ── ImageNet normalisation (matches training preprocessing) ────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── GTSRB class names (43 classes, indices 0–42) ──────────────────────────────
CLASS_NAMES = [
    "speed_limit_20",    "speed_limit_30",    "speed_limit_50",
    "speed_limit_60",    "speed_limit_70",    "speed_limit_80",
    "end_speed_limit_80","speed_limit_100",   "speed_limit_120",
    "no_passing",        "no_passing_heavy",  "right_of_way_ahead",
    "priority_road",     "yield",             "stop",
    "no_vehicles",       "no_heavy_vehicles", "no_entry",
    "general_caution",   "curve_left",        "curve_right",
    "double_curve",      "bumpy_road",        "slippery_road",
    "road_narrows",      "road_work",         "traffic_signals",
    "pedestrians",       "children_crossing", "bicycles_crossing",
    "ice_snow",          "wild_animals",      "end_restrictions",
    "turn_right_ahead",  "turn_left_ahead",   "ahead_only",
    "go_straight_right", "go_straight_left",  "keep_right",
    "keep_left",         "roundabout",        "end_no_passing",
    "end_no_passing_heavy",
]

# ── Indici clase după formă — folosiți pentru shape-based correction ──────────
# Semne CIRCULARE (limitele de viteză, interdicții):
_CIRCLE_CLASSES  = {i for i, n in enumerate(CLASS_NAMES)
                    if n.startswith("speed_limit") or n in
                    ("no_passing", "no_passing_heavy", "no_vehicles",
                     "no_heavy_vehicles", "no_entry", "end_speed_limit_80",
                     "end_restrictions", "end_no_passing", "end_no_passing_heavy")}
# Semne TRIUNGHIULARE (avertismente, cedează):
_TRIANGLE_CLASSES = {i for i, n in enumerate(CLASS_NAMES)
                     if n in ("yield", "general_caution", "curve_left", "curve_right",
                               "double_curve", "bumpy_road", "slippery_road",
                               "road_narrows", "road_work", "traffic_signals",
                               "pedestrians", "children_crossing", "bicycles_crossing",
                               "ice_snow", "wild_animals", "right_of_way_ahead")}
# Semne OCTOGONALE (stop) sau dreptunghiulare (prioritate):
_OCTAGON_CLASSES  = {CLASS_NAMES.index("stop"), CLASS_NAMES.index("priority_road")}

# ── Semnele fizice disponibile pe pistă ────────────────────────────────────────
# STOP, CEDEAZĂ (yield), 30, 50, 90, 100
#
# ⚠️ GTSRB nu are clasa "90 km/h" — trece direct 80→100.
#    speed_limit_80 este remapat să se comporte ca "90 km/h" pe pista noastră.
#    Dacă ai semn fizic de 80, înlocuiește maparea de mai jos.

# ── Which classes map to which driving behaviour label ─────────────────────────
_BEHAVIOUR_MAP = {
    "stop":              "stop",
    "yield":             "yield",
    "no_entry":          "stop",
    # Limitele de viteză de pe pistă
    "speed_limit_30":    "speed_limit_30",
    "speed_limit_50":    "speed_limit_50",
    "speed_limit_80":    "speed_limit_90",   # 80 remapat → 90 (nu există 90 în GTSRB)
    "speed_limit_100":   "speed_limit_100",
    # Restul claselor GTSRB — ignorate (None = nu triggerează comportament)
    "speed_limit_20":    None,
    "speed_limit_60":    None,
    "speed_limit_70":    None,
    "speed_limit_120":   None,
    "end_speed_limit_80":"end_restrictions",
    "end_restrictions":  "end_restrictions",
    "traffic_signals":   "traffic_signals",
}

# Speed-limit behaviour label → multiplicator aplicat pe base_speed
# Valorile sunt mari intenționat — base_speed e deja mic (ex. 0.32),
# multiplicatorul nu trebuie să-l mai reducă mult ca să nu oprească mașina.
_SPEED_MULT = {
    "speed_limit_30":  0.78,   # ~25% din max  → cel mai lent, dar merge
    "speed_limit_50":  0.88,   # ~28% din max  → mediu
    "speed_limit_90":  0.96,   # ~31% din max  → aproape full
    "speed_limit_100": 1.00,   # full speed
}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — colour-based candidate detector
# ══════════════════════════════════════════════════════════════════════════════

# CLAHE instance for sign ROI enhancement
_clahe_sign = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))


def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the LAB L-channel to improve contrast for bad cameras."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe_sign.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _get_sign_shape(cnt) -> str:
    """
    Determină forma semnului din contur.
    Returnează: 'circle' | 'triangle' | 'octagon' | 'other'

    Logica:
      - Circularitate ridicată (≥0.65)  → cerc   → limitele de viteză, interdicții
      - Triunghi (≈3 laturi)            → triunghi → cedează, avertismente
      - Poligon cu ≥7 laturi            → octogon  → stop
    """
    perimeter = cv2.arcLength(cnt, True)
    area      = cv2.contourArea(cnt)
    if perimeter < 1:
        return 'other'
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity >= 0.62:
        return 'circle'
    approx  = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    n_sides = len(approx)
    if n_sides == 3:
        return 'triangle'
    if n_sides >= 7:
        return 'octagon'
    return 'other'


def _find_candidates(frame: np.ndarray) -> list:
    """
    Returns a list of (x, y, w, h, shape) for regions that have
    the colour signature of a traffic sign (red, blue, yellow, or white).
    `shape` ∈ {'circle', 'triangle', 'octagon', 'other'}.
    Applies CLAHE first to compensate for bad/dark cameras.
    """
    # Only search the top 60% of the frame — signs are on the sides/ahead,
    # never on the floor where the lane tape lives.
    fh_full = frame.shape[0]
    search_h = int(fh_full * 0.60)
    frame = frame[:search_h]

    enhanced = _enhance_frame(frame)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    # Red (wraps around 0° in hue) — S≥70 avoids pale skin/orange noise
    red_lo = cv2.inRange(hsv, (0,   70, 60), (12,  255, 255))
    red_hi = cv2.inRange(hsv, (158, 70, 60), (179, 255, 255))
    red    = cv2.bitwise_or(red_lo, red_hi)

    # Blue (road-sign blue)
    blue   = cv2.inRange(hsv, (95,  60, 50), (140, 255, 255))

    # Yellow removed: all signs on the track (stop, yield, speed limits) are
    # red/white.  Yellow catches lane tape and triggers false YIELD detections.

    # White mask removed: causes massive false positives (sky, walls, paper).
    # Speed-limit / stop / yield signs are all caught by their red border.

    mask = red | blue

    # Clean up noise + bridge gaps in red RINGS (speed-limit borders).
    # CLOSE iter=2 keeps a thin red ring as one solid circular contour instead
    # of breaking into arcs, but stays small enough NOT to merge adjacent signs.
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    seen_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_REGION_PX:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if not (0.4 < aspect < 2.5):
            continue
        shape = _get_sign_shape(cnt)
        # Padding: 15% of bbox size
        pad = int(max(w, h) * 0.15)
        fh, fw = frame.shape[:2]
        x = max(0, x - pad);  y = max(0, y - pad)
        w = min(fw - x, w + 2 * pad)
        h = min(fh - y, h + 2 * pad)
        # Skip near-duplicate boxes (overlap > 80%)
        duplicate = False
        for (bx, by, bw, bh, _) in seen_boxes:
            ix = max(0, min(x+w, bx+bw) - max(x, bx))
            iy = max(0, min(y+h, by+bh) - max(y, by))
            if ix * iy > 0.8 * min(w*h, bw*bh):
                duplicate = True; break
        if not duplicate:
            candidates.append((x, y, w, h, shape))
            seen_boxes.append((x, y, w, h, shape))

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — ONNX classifier
# ══════════════════════════════════════════════════════════════════════════════

class SignDetector:
    """
    Detects and classifies traffic signs in a frame using:
      1. Fast colour segmentation to find candidate regions.
      2. Custom MobileNetV2 ONNX model (trained on GTSRB) to classify each region.

    Interface is identical to the previous YOLOv8-based detector:
      detector = SignDetector()
      detections = detector.detect(frame)
      # → [{'label': str, 'confidence': float, 'bbox': (x,y,w,h), 'area_frac': float}, ...]
    """

    def __init__(self, model_path: str = ONNX_MODEL_PATH):
        self._net  = None
        self._lock = threading.Lock()
        if os.path.exists(model_path):
            try:
                self._net = cv2.dnn.readNetFromONNX(model_path)
                # Use CPU backend — works on every Pi without extra setup
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"[signs] ONNX model loaded: {model_path}")
            except Exception as exc:
                print(f"[signs] failed to load ONNX model: {exc}")
        else:
            print(f"[signs] model not found at {model_path}")
            print("[signs]  Train it: cd training && python train.py && python export.py")
            print(f"[signs]  Then copy checkpoints/traffic_sign_classifier.onnx to {_HERE}")

    @property
    def available(self) -> bool:
        return self._net is not None

    def detect(self, frame: np.ndarray) -> list:
        """
        Returns a list of dicts:
        {
          'label':      str,    # GTSRB class name, e.g. 'stop', 'yield'
          'confidence': float,
          'bbox':       (x, y, w, h),
          'area_frac':  float,  # fraction of frame area — proxy for distance
        }
        """
        if self._net is None:
            return []

        fh, fw = frame.shape[:2]
        frame_area = fh * fw
        candidates = _find_candidates(frame)
        detections = []

        for (x, y, w, h, shape) in candidates:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            label, conf = self._classify(roi, shape)
            if conf < CONF_THRESHOLD:
                continue

            area_frac = (w * h) / frame_area
            detections.append({
                "label":      label,
                "confidence": round(conf, 3),
                "bbox":       (x, y, x + w, y + h),  # x1,y1,x2,y2 for annotate_frame
                "area_frac":  round(area_frac, 4),
                "shape":      shape,
            })

        # Non-maximum suppression across overlapping boxes
        if len(detections) > 1:
            detections = _nms(detections)

        return detections

    def _classify(self, roi: np.ndarray, shape: str = 'other') -> tuple:
        """
        Returns (class_name, confidence).
        Rulează clasificatorul pe ROI normal + CLAHE-enhanced, alege cel mai sigur.
        Apoi aplică shape-based correction:
          - 'circle'   → rezultatul trebuie să fie o clasă circulară (limită de viteză etc.)
                         Dacă nu e, ia cel mai bun scor din clasele circulare.
          - 'triangle' → rezultatul trebuie să fie o clasă triunghiulară.
          - 'octagon'  → rezultatul trebuie să fie stop / priority_road.
        Asta corectează confuzia frecventă yield↔speed_limit.
        """
        results = []
        all_probs = []
        for enhanced in (roi, _enhance_frame(roi)):
            img = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            img = (img - _MEAN) / _STD
            blob = img.transpose(2, 0, 1)[np.newaxis]   # HWC → NCHW
            with self._lock:
                self._net.setInput(blob)
                logits = self._net.forward()[0]          # shape (43,)
            probs = _softmax(logits)
            top1  = int(np.argmax(probs))
            results.append((CLASS_NAMES[top1], float(probs[top1])))
            all_probs.append(probs)

        # Alege cea mai sigură predicție brută
        best_label, best_conf = max(results, key=lambda r: r[1])
        best_probs = all_probs[0] if results[0][1] >= results[1][1] else all_probs[1]

        # ── Shape-based correction ─────────────────────────────────────────────
        top1_idx = CLASS_NAMES.index(best_label)

        # ── STOP / semn rotund umplut cu roșu ──────────────────────────────────
        # Un octogon (STOP) e citit ca 'circle' (circularitate ~0.95).  Diferența
        # față de o limită de viteză: STOP e UMPLUT cu roșu, limita e un INEL roșu
        # cu centru alb.  Măsurăm fracția de roșu → dacă e mare, e STOP/no_entry,
        # indiferent ce zice clasificatorul (prinde STOP-ul chiar dacă modelul
        # e nesigur sau îl confundă cu o limită de viteză).
        hsv_roi  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_roi  = cv2.bitwise_or(
            cv2.inRange(hsv_roi, (0,   70, 60), (12,  255, 255)),
            cv2.inRange(hsv_roi, (158, 70, 60), (179, 255, 255)))
        red_fill = float(np.count_nonzero(red_roi)) / max(1, red_roi.size)

        if shape in ('circle', 'octagon') and red_fill >= RED_FILL_STOP:
            stop_i = CLASS_NAMES.index("stop")
            ne_i   = CLASS_NAMES.index("no_entry")
            pick   = stop_i if best_probs[stop_i] >= best_probs[ne_i] else ne_i
            # forțăm încrederea să treacă pragul — un octogon plin de roșu e STOP
            return CLASS_NAMES[pick], max(float(best_probs[pick]), 0.62)

        # ── YIELD/triunghi CER un contur TRIANGULAR ────────────────────────────
        # Cauza „yield random": modelul scotea o clasă triunghi pentru orice pată
        # roșie. Acum, dacă prezice triunghi dar conturul NU e triunghi → e fals.
        if top1_idx in _TRIANGLE_CLASSES and shape != 'triangle':
            if shape == 'circle':
                # probabil o limită de viteză citită greșit ca yield → ia cercul
                circle_scores = {i: best_probs[i] for i in _CIRCLE_CLASSES}
                best_circle   = max(circle_scores, key=circle_scores.get)
                if best_probs[best_circle] >= 0.30:
                    return CLASS_NAMES[best_circle], max(0.0, float(best_probs[best_circle]) - 0.05)
            return best_label, 0.0   # suprimă yield-ul fals (sub prag → filtrat)

        # Contur triunghi, dar modelul a zis cerc → forțează triunghi (yield real)
        if shape == 'triangle' and top1_idx in _CIRCLE_CLASSES:
            triangle_scores = {i: best_probs[i] for i in _TRIANGLE_CLASSES}
            best_tri        = max(triangle_scores, key=triangle_scores.get)
            if best_probs[best_tri] >= 0.30:
                best_label = CLASS_NAMES[best_tri]
                best_conf  = max(0.0, float(best_probs[best_tri]) - 0.05)

        return best_label, best_conf


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _nms(detections: list, iou_threshold: float = 0.4) -> list:
    """Greedy NMS: keep highest-confidence box when IoU > threshold."""
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in detections:
        if all(_iou(det["bbox"], k["bbox"]) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union


# ══════════════════════════════════════════════════════════════════════════════
# Drive behaviour state machine
# ══════════════════════════════════════════════════════════════════════════════

class DriveBehavior:
    """
    Traduce detecțiile de semne în multiplicator de viteză (0.0 – 1.0).

    Tabel comportamente (tunat pentru pistă mică):
    ──────────────────────────────────────────────
    STOP / no_entry  → încetinire 0.25× (0.8 s) → oprire completă (2.5 s) → reia
    CEDEAZĂ (yield)  → încetinire 0.30× (1.5 s) → reia
    30 km/h          → cap persistent 0.30×
    50 km/h          → cap persistent 0.50×
    90 km/h          → cap persistent 0.90×  (detectat ca speed_limit_80 în GTSRB)
    100 km/h         → cap persistent 1.00× (full speed)
    end_restrictions → șterge cap → reia viteza completă
    """

    _STATES = {
        "normal":         1.00,
        "slowing_stop":   0.25,
        "stopped_sign":   0.00,
        "slowing_yield":  0.30,
        "slowing_caution":0.35,
    }

    def __init__(self):
        self.state        = "normal"
        self.multiplier   = 1.0
        self.sign         = None
        self._speed_cap   = 1.0    # limită de viteză persistentă
        self._timer       = None
        self._counts      = defaultdict(int)

    def update(self, detections: list) -> tuple:
        """
        Apelat o dată per ciclu de detecție.
        Returnează (multiplicator_efectiv, eticheta_semn_activ).
        """
        # Traduce class names → behaviour labels
        close_behaviours = set()
        for d in detections:
            if d["area_frac"] >= TRIGGER_AREA:
                beh = _BEHAVIOUR_MAP.get(d["label"])
                if beh:
                    close_behaviours.add(beh)

        # Contoare de confirmare (evită false positive-uri)
        for beh in list(self._counts):
            self._counts[beh] = self._counts[beh] + 1 if beh in close_behaviours else 0
        for beh in close_behaviours:
            if beh not in self._counts:
                self._counts[beh] = 1

        confirmed = {beh for beh, n in self._counts.items() if n >= CONFIRM_FRAMES}

        # ── Limite de viteză persistente ──────────────────────────────────────
        for beh in confirmed:
            if beh in _SPEED_MULT:
                cap = _SPEED_MULT[beh]
                if cap != self._speed_cap:
                    self._speed_cap = cap
                    self.sign = beh
                    # Afișare frumoasă: speed_limit_90 → "90 km/h"
                    kmh = beh.replace("speed_limit_", "")
                    print(f"[behavior] ⚡ limită viteză → {kmh} km/h  (×{cap:.0%})")

        if "end_restrictions" in confirmed:
            self._speed_cap = 1.0
            if self.state == "normal":
                self.sign = None
            print("[behavior] ✅ limită ridicată — full speed")

        # ── Mașina de stări ───────────────────────────────────────────────────
        if self.state == "normal":
            if "stop" in confirmed:
                self._go("slowing_stop", "stop")
                self._after(0.8, self._full_stop)      # 0.8 s încetinire → stop
                print("[behavior] 🛑 STOP detectat — oprire")
            elif "yield" in confirmed:
                self._go("slowing_yield", "yield")
                self._after(1.5, self.resume)           # 1.5 s cedează → reia
                print("[behavior] ⚠️  CEDEAZĂ detectat — încetinire")
            elif "traffic_signals" in confirmed:
                self._go("slowing_caution", "traffic_signals")
                self._after(2.0, self.resume)

        # Multiplicator efectiv = min(stare mașină, cap limită viteză)
        eff = min(self.multiplier, self._speed_cap)
        return eff, self.sign

    def resume(self):
        """Force-return to normal (call on mode switch or green light)."""
        if self._timer:
            self._timer.cancel()
        self._go("normal", None)
        self._counts.clear()

    def _go(self, state, sign):
        self.state      = state
        self.multiplier = self._STATES.get(state, 1.0)
        if sign is not None:
            self.sign = sign

    def _full_stop(self):
        self._go("stopped_sign", "stop")
        self._after(2.5, self.resume)   # oprire 2.5 s → reia

    def _after(self, delay, fn):
        if self._timer:
            self._timer.cancel()
        t = threading.Timer(delay, fn)
        t.daemon = True
        t.start()
        self._timer = t


# ══════════════════════════════════════════════════════════════════════════════
# Frame annotation
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "stop":              (30,  30, 230),
    "yield":             (0,  150, 255),
    "no_entry":          (30,  30, 230),
    "traffic_signals":   (0,  200, 255),
    "road_work":         (0,  165, 255),
    "general_caution":   (0,  200, 255),
}
# Speed-limit signs → green tones
for _i, _n in enumerate(CLASS_NAMES):
    if _n.startswith("speed_limit"):
        _COLORS[_n] = (0, 200 - _i*3, 100 + _i*3)


def annotate_frame(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes + class labels on *frame* (in-place) and return it."""
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        lbl   = d["label"]
        color = _COLORS.get(lbl, (180, 180, 60))
        shape_icon = {"circle": "○", "triangle": "△", "octagon": "⬡", "other": "?"}.get(d.get("shape","other"), "?")
        text  = f"{shape_icon} {lbl.replace('_',' ').upper()}  {d['confidence']:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, cv2.FILLED)
        cv2.putText(frame, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
