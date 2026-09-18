# Anexe — Cod sursă (extrase relevante)

> Mașină autonomă pe Raspberry Pi 4. Acest document conține fragmentele de cod
> cele mai importante din proiect, organizate pe module. Comentariile din cod
> sunt păstrate ca în sursă.

**Cuprins anexe**
- Anexa A — Regulator PID (`pid.py`)
- Anexa B — Control motoare (`MotorModule.py`)
- Anexa C — Array senzori ultrasonici (`UltrasonicModule.py`)
- Anexa D — Detecția benzii (`app.py::_detect_lane`)
- Anexa E — Bucla de control Lane Assist (`app.py::_lane_assist_thread`)
- Anexa F — Frânare de urgență ultrasonică (`app.py::_ultrasonic_thread`)
- Anexa G — Detecția semnelor: candidați + clasificare (`sign_detection.py`)
- Anexa H — Comportament la semne (`sign_detection.py::DriveBehavior`)
- Anexa I — Detecția obstacolelor (`obstacle_detection.py`)
- Anexa J — Parcare automată (`parking_module.py`)
- Anexa K — Funcții auxiliare de viziune (`utlis.py`)

---

## Anexa A — Regulator PID thread-safe (`pid.py`)

PID-ul transformă eroarea de bandă (curba ∈ [-1, 1]) în comanda de viraj. Include
anti-windup pe integrator și filtru EMA pe derivată (suprimă zgomotul senzorului).

```python
import time
import threading


class PIDController:
    """Discrete-time PID with derivative filtering and anti-windup."""

    def __init__(self, kp: float = 0.7, ki: float = 0.02, kd: float = 0.15):
        self._lock = threading.Lock()
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd)
        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0   # EMA-filtered derivative
        self._prev_time  = None
        self.last_p = self.last_i = self.last_d = 0.0
        self.last_output = self.last_error = 0.0

    def compute(self, error: float) -> float:
        """Call once per lane-detect cycle. error ∈ [-1, 1] → turn ∈ [-1, 1]."""
        now = time.monotonic()
        with self._lock:
            # dt — clamped to realistic range (5 ms … 200 ms)
            if self._prev_time is None:
                dt = 0.05
            else:
                dt = max(0.005, min(now - self._prev_time, 0.20))
            self._prev_time = now

            # Proportional
            p = self.kp * error
            # Integral with anti-windup clamp
            self._integral += error * dt
            self._integral  = max(-1.5, min(1.5, self._integral))
            i = self.ki * self._integral
            # Derivative with EMA low-pass (α = 0.9)
            raw_d = (error - self._prev_error) / dt
            self._d_filtered = 0.9 * self._d_filtered + 0.1 * raw_d
            d = self.kd * self._d_filtered
            self._prev_error = error
            # Sum + clamp
            output = max(-1.0, min(1.0, p + i + d))

            self.last_error  = round(error,  4)
            self.last_p = round(p, 4); self.last_i = round(i, 4)
            self.last_d = round(d, 4); self.last_output = round(output, 4)
        return output

    def reset(self):
        """Zero integrator and derivative state. Call on mode switches."""
        with self._lock:
            self._integral = self._prev_error = self._d_filtered = 0.0
            self._prev_time = None
            self.last_p = self.last_i = self.last_d = 0.0
            self.last_output = self.last_error = 0.0

    def set_gains(self, kp=None, ki=None, kd=None):
        with self._lock:
            if kp is not None: self.kp = max(0.0, float(kp))
            if ki is not None: self.ki = max(0.0, float(ki))
            if kd is not None: self.kd = max(0.0, float(kd))
```

---

## Anexa B — Control motoare (`MotorModule.py`)

Mașina e skid-steer (4 motoare DC). `move(speed, turn)` aplică control diferențial:
`left = speed + turn`, `right = speed - turn`. Astfel se poate roti și pe loc.

```python
from gpiozero import Motor, Device
from gpiozero.pins.rpigpio import RPiGPIOFactory

Device.pin_factory = RPiGPIOFactory()

front_left  = Motor(forward=17, backward=27)
front_right = Motor(forward=22, backward=23)
rear_left   = Motor(forward=5,  backward=6)
rear_right  = Motor(forward=13, backward=19)


def stop_motors():
    front_left.stop(); rear_left.stop()
    front_right.stop(); rear_right.stop()


class MotorModule:
    def move(self, speed=0.0, turn=0.0, t=0):
        """speed: -1.0 (reverse) … +1.0 (forward);  turn: -1.0 (left) … +1.0 (right)"""
        left  = max(min(speed + turn, 1.0), -1.0)
        right = max(min(speed - turn, 1.0), -1.0)

        # LEFT side = front_left + rear_left
        if left > 0:
            front_left.forward(left);  rear_left.forward(left)
        elif left < 0:
            front_left.backward(abs(left)); rear_left.backward(abs(left))
        else:
            front_left.stop(); rear_left.stop()

        # RIGHT side = front_right + rear_right
        if right > 0:
            front_right.forward(right); rear_right.forward(right)
        elif right < 0:
            front_right.backward(abs(right)); rear_right.backward(abs(right))
        else:
            front_right.stop(); rear_right.stop()

        if t:
            import time; time.sleep(t); self.stop()

    def stop(self):
        stop_motors()
```

---

## Anexa C — Array de 6 senzori ultrasonici (`UltrasonicModule.py`)

3 senzori față + 3 spate. `partial=True` este critic: fără el, `.distance` se
blochează când un senzor nu primește ecou (cablaj/GND), înghețând firul de execuție.

```python
from gpiozero import DistanceSensor, Device
from gpiozero.pins.rpigpio import RPiGPIOFactory
import threading

Device.pin_factory = RPiGPIOFactory()

SENSOR_PINS = {
    "front_left":   {"trig": 4,  "echo": 24},
    "front_center": {"trig": 12, "echo": 25},
    "front_right":  {"trig": 16, "echo": 26},
    "rear_left":    {"trig": 20, "echo": 21},
    "rear_center":  {"trig": 18, "echo": 8},
    "rear_right":   {"trig": 7,  "echo": 9},
}
MAX_DISTANCE_M = 3.0


class UltrasonicArray:
    """Gestionează 6 senzori HC-SR04 simultan (non-blocking via gpiozero)."""

    def __init__(self):
        self._sensors = {}
        self._lock = threading.Lock()
        for name, pins in SENSOR_PINS.items():
            try:
                s = DistanceSensor(
                    echo=pins["echo"], trigger=pins["trig"],
                    max_distance=MAX_DISTANCE_M,
                    queue_len=3,     # medie pe 3 citiri → stabil dar rapid
                    partial=True,    # CRITIC: .distance NU mai blochează fără ecou
                )
                self._sensors[name] = s
                print(f"[Ultrasonic] ✓ {name:14s} TRIG={pins['trig']:2d} ECHO={pins['echo']:2d}")
            except Exception as e:
                print(f"[Ultrasonic] ✗ {name:14s} — {e}")

        ok, total = len(self._sensors), len(SENSOR_PINS)
        print(f"[Ultrasonic] {ok}/{total} senzori inițializați")
        if ok == 0:
            raise RuntimeError("niciun senzor HC-SR04 inițializat — verifică cablajul")

    @property
    def count(self) -> int:
        return len(self._sensors)

    def get_distances(self) -> dict:
        """Distanțele curente în cm pentru toți senzorii (None la eroare)."""
        result = {}
        for name, sensor in self._sensors.items():
            try:
                result[name] = round(sensor.distance * 100, 1)
            except Exception:
                result[name] = None
        return result

    def front_distances(self) -> dict:
        return {k: self.get_distances().get(k)
                for k in ("front_left", "front_center", "front_right")}


_array = None

def get_array() -> UltrasonicArray:
    """Singleton folosit din app.py."""
    global _array
    if _array is None:
        _array = UltrasonicArray()
    return _array
```

---

## Anexa D — Detecția benzii (`app.py::_detect_lane`)

Pipeline fără warp: ROI jos + mască galbenă + **centroid ponderat** per jumătate
+ **netezire temporală** (median → EMA → limită de pantă) ce elimină „spasmele".

```python
def _detect_lane(img):
    """Returns (annotated_frame, curve, confidence). curve ∈ [-1,1]."""
    global _curve_list, _lane_curve_smooth, _lane_raw_hist
    h_orig, w_orig = img.shape[:2]

    # 1. ROI: 60% de jos al imaginii
    roi_start = int(h_orig * 0.40)
    roi_proc  = cv2.resize(img[roi_start:, :], (320, 120))
    rh, rw = roi_proc.shape[:2]; half = rw // 2

    # 2. Mască galbenă HSV
    blurred = cv2.GaussianBlur(roi_proc, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, np.array([10, 100, 80]), np.array([45, 255, 255]))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    yellow_px = int(np.count_nonzero(mask))

    # 3. Centroid PONDERAT după densitatea de galben per coloană (robust la zgomot)
    col_l = mask[:, :half].sum(axis=0).astype(np.float32)
    col_r = mask[:, half:].sum(axis=0).astype(np.float32)
    MIN_SIDE = 255.0 * 10   # min ~10 px galbeni pe acea parte
    left_x  = int((np.arange(half)     * col_l).sum() / col_l.sum()) if col_l.sum() >= MIN_SIDE else None
    right_x = int((np.arange(half, rw) * col_r).sum() / col_r.sum()) if col_r.sum() >= MIN_SIDE else None

    # 4. Centrul benzii (+ confidence)
    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) // 2; confidence = 100
    elif left_x is not None:
        lane_center = left_x + half // 2;      confidence = 40
    elif right_x is not None:
        lane_center = right_x - half // 2;     confidence = 40
    else:
        lane_center = half;                    confidence = 0

    # 5. Eroare + boost pe o singură linie + deadband
    rawCurve = lane_center - half
    if confidence == 40:
        rawCurve = int(rawCurve * 2.0)
    if abs(rawCurve) < 3:
        rawCurve = 0

    # 6. Normalizare agresivă /35
    raw_curve = max(-1.0, min(1.0, rawCurve / 35.0)) if yellow_px > 30 else 0.0

    # 7. Netezire temporală (anti-spazz): median-3 → EMA → limită de pantă
    _lane_raw_hist.append(raw_curve)
    if len(_lane_raw_hist) > LANE_MEDIAN_N:
        _lane_raw_hist.pop(0)
    med  = sorted(_lane_raw_hist)[len(_lane_raw_hist) // 2]          # respinge vârf de 1 cadru
    ema  = LANE_SMOOTH_ALPHA * _lane_curve_smooth + (1 - LANE_SMOOTH_ALPHA) * med
    step = max(-LANE_MAX_STEP, min(LANE_MAX_STEP, ema - _lane_curve_smooth))
    _lane_curve_smooth = max(-1.0, min(1.0, _lane_curve_smooth + step))
    curve = _lane_curve_smooth

    # (urmează desenarea overlay-ului pe cadru — omisă aici)
    return annotated, curve, confidence
```

---

## Anexa E — Bucla de control Lane Assist (`app.py::_lane_assist_thread`)

Firul de 20 Hz: detecție bandă → PID → adaptive speed → respectare limită de
viteză → blocare la obstacol/coliziune → comandă motoare.

```python
# ── PID turn correction ──
turn = pid.compute(curve) if use_pid else max(-MAX_TURN, min(MAX_TURN, curve))
turn = max(-MAX_TURN, min(MAX_TURN, turn))
spd  = current_speed

# ── Adaptive speed: încetinește în curbă ──
# adap = 1 - c_k*|curve|  (c_k=0.85 → în curbă strânsă viteza ~0)
adap_factor = max(0.0, 1.0 - c_k * abs(curve)) if use_adap else 1.0

# ── Blochează motoarele dacă e obstacol sau coliziune ultrasonică ──
with obstacle_lock:
    obs_dangerous = latest_obstacle.get("dangerous", False)

if (not obs_dangerous or not obstacles_enabled) and not _ultrasonic_stopped:
    # ── Respectarea limitei de viteză: mașina merge LA viteza din semn ──
    base_spd = spd
    with speed_limit_lock:
        sl_frac = active_speed_limit_frac
        sl_on   = speed_limit_enabled
    if sl_on and sl_frac is not None:
        base_spd = sl_frac          # înlocuiește slider-ul (obey, nu doar plafon)
    effective_speed = base_spd * drive_behavior.multiplier * adap_factor
    with motor_lock:
        motor.move(speed=effective_speed, turn=turn)
else:
    effective_speed = 0.0; turn = 0.0   # oprit

# telemetrie live către browser (WebSocket)
socketio.emit("telemetry", {"curve": round(curve, 3),
                            "speed": round(effective_speed, 2),
                            "turn":  round(turn, 2), ...})
```

---

## Anexa F — Frânare de urgență ultrasonică (`app.py::_ultrasonic_thread`)

Firul de 15 Hz folosește cei 3 senzori din **față**. Dacă cel mai apropiat <30 cm
→ oprire imediată (și blochează lane assist până se eliberează calea).

```python
def _ultrasonic_thread():
    global ultrasonic_cm, ultrasonic_all, _ultrasonic_stopped
    if _ultra is None:
        return
    while True:
        time.sleep(1.0 / 15)
        dists  = _ultra.get_distances()
        fronts = [dists.get(k) for k in ("front_left", "front_center", "front_right")]
        front_vals = [v for v in fronts if v is not None]
        front_min  = min(front_vals) if front_vals else None

        with ultrasonic_lock:
            enabled = ultrasonic_enabled
            ultrasonic_all = dists; ultrasonic_cm = front_min

        if _parking_active():        # parcarea are propria siguranță
            socketio.emit("ultrasonic_update", {...}); continue

        dangerous = enabled and front_min is not None and front_min < CRASH_DISTANCE_CM
        if dangerous and not _ultrasonic_stopped:
            _ultrasonic_stopped = True
            with motor_lock:
                motor.stop()
            print(f"[ultrasonic] *** CRASH STOP — {front_min} cm ***")
        elif not dangerous and _ultrasonic_stopped:
            _ultrasonic_stopped = False

        socketio.emit("ultrasonic_update", {
            "cm": front_min, "dangerous": dangerous,
            "enabled": enabled, "distances": dists})
```

---

## Anexa G — Detecția semnelor: candidați + clasificare (`sign_detection.py`)

**Etapa 1** găsește regiuni roșii/albastre în top 60% din cadru. **Etapa 2**
clasifică fiecare candidat cu modelul ONNX și aplică corecțiile shape + red-fill.

```python
def _find_candidates(frame):
    """Returnează (x, y, w, h, shape) pentru regiuni cu culoare de semn."""
    # Caută doar în top 60% din cadru (jos e pista)
    frame = frame[:int(frame.shape[0] * 0.60)]
    hsv = cv2.cvtColor(_enhance_frame(frame), cv2.COLOR_BGR2HSV)  # CLAHE întâi

    red = cv2.bitwise_or(cv2.inRange(hsv, (0,   70, 60), (12,  255, 255)),
                         cv2.inRange(hsv, (158, 70, 60), (179, 255, 255)))
    blue = cv2.inRange(hsv, (95, 60, 50), (140, 255, 255))
    mask = red | blue   # galbenul/albul eliminate (false positive pe bandă/perete)

    # CLOSE iter=2 → unește inelul roșu al limitelor de viteză într-un contur plin
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_REGION_PX:          # ignoră blob-uri mici
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (0.4 < w / max(h, 1) < 2.5):               # semnele sunt ~pătrate
            continue
        shape = _get_sign_shape(cnt)                       # circle/triangle/octagon
        pad = int(max(w, h) * 0.15)                        # padding 15%
        x = max(0, x - pad); y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2*pad); h = min(frame.shape[0] - y, h + 2*pad)
        candidates.append((x, y, w, h, shape))
    return candidates


def detect(self, frame):
    """Returnează listă de {label, confidence, bbox, area_frac, shape}."""
    if self._net is None:
        return []
    frame_area = frame.shape[0] * frame.shape[1]
    detections = []
    for (x, y, w, h, shape) in _find_candidates(frame):
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        label, conf = self._classify(roi, shape)
        if conf < CONF_THRESHOLD:                          # 0.60
            continue
        detections.append({"label": label, "confidence": round(conf, 3),
                           "bbox": (x, y, x+w, y+h),
                           "area_frac": round((w*h) / frame_area, 4), "shape": shape})
    if len(detections) > 1:
        detections = _nms(detections)                      # non-maximum suppression
    return detections
```

**Clasificare + corecții inteligente** (cheia robusteții fără reantrenare):

```python
def _classify(self, roi, shape='other'):
    # Rulează modelul pe ROI normal + CLAHE, alege predicția cea mai sigură
    results, all_probs = [], []
    for enhanced in (roi, _enhance_frame(roi)):
        img = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD                          # normalizare ImageNet
        blob = img.transpose(2, 0, 1)[np.newaxis]           # HWC → NCHW
        with self._lock:
            self._net.setInput(blob)
            logits = self._net.forward()[0]                 # 43 clase
        probs = _softmax(logits)
        results.append((CLASS_NAMES[int(np.argmax(probs))], float(probs.max())))
        all_probs.append(probs)

    best_label, best_conf = max(results, key=lambda r: r[1])
    best_probs = all_probs[0] if results[0][1] >= results[1][1] else all_probs[1]
    top1_idx = CLASS_NAMES.index(best_label)

    # ── STOP prin fracția de roșu ──
    # Un octogon are circularitate ~0.95 → citit ca 'circle'. STOP e UMPLUT cu roșu,
    # limita de viteză e un INEL roșu cu centru alb → fracție de roșu mică.
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_roi = cv2.bitwise_or(cv2.inRange(hsv_roi, (0,70,60),(12,255,255)),
                             cv2.inRange(hsv_roi, (158,70,60),(179,255,255)))
    red_fill = float(np.count_nonzero(red_roi)) / max(1, red_roi.size)
    if shape in ('circle', 'octagon') and red_fill >= RED_FILL_STOP:   # 0.30
        stop_i = CLASS_NAMES.index("stop"); ne_i = CLASS_NAMES.index("no_entry")
        pick = stop_i if best_probs[stop_i] >= best_probs[ne_i] else ne_i
        return CLASS_NAMES[pick], max(float(best_probs[pick]), 0.62)

    # ── Anti-fals YIELD: clasa triunghi acceptată DOAR dacă și conturul e triunghi ──
    if top1_idx in _TRIANGLE_CLASSES and shape != 'triangle':
        if shape == 'circle':                               # probabil limită citită ca yield
            best_circle = max(_CIRCLE_CLASSES, key=lambda i: best_probs[i])
            if best_probs[best_circle] >= 0.30:
                return CLASS_NAMES[best_circle], max(0.0, float(best_probs[best_circle]) - 0.05)
        return best_label, 0.0                              # suprimă (sub prag → filtrat)

    # Contur triunghi, dar model a zis cerc → forțează triunghi (yield real)
    if shape == 'triangle' and top1_idx in _CIRCLE_CLASSES:
        best_tri = max(_TRIANGLE_CLASSES, key=lambda i: best_probs[i])
        if best_probs[best_tri] >= 0.30:
            best_label = CLASS_NAMES[best_tri]
            best_conf  = max(0.0, float(best_probs[best_tri]) - 0.05)

    return best_label, best_conf
```

---

## Anexa H — Comportament la semne (`sign_detection.py::DriveBehavior`)

Mașină de stări care traduce detecțiile în multiplicator de viteză. Necesită
confirmare pe mai multe cadre (anti-fals) și o suprafață minimă (proxy distanță).

```python
class DriveBehavior:
    _STATES = {"normal": 1.00, "slowing_stop": 0.25, "stopped_sign": 0.00,
               "slowing_yield": 0.30, "slowing_caution": 0.35}

    def __init__(self):
        self.state = "normal"; self.multiplier = 1.0; self.sign = None
        self._speed_cap = 1.0; self._timer = None
        self._counts = defaultdict(int)

    def update(self, detections):
        """Apelat o dată per ciclu. Returnează (multiplicator_efectiv, semn_activ)."""
        # Traduce clasele apropiate (area_frac ≥ TRIGGER_AREA) în comportamente
        close = set()
        for d in detections:
            if d["area_frac"] >= TRIGGER_AREA:
                beh = _BEHAVIOUR_MAP.get(d["label"])
                if beh:
                    close.add(beh)

        # Contoare de confirmare (evită false positive-uri)
        for beh in list(self._counts):
            self._counts[beh] = self._counts[beh] + 1 if beh in close else 0
        for beh in close:
            self._counts.setdefault(beh, 1)
        confirmed = {b for b, n in self._counts.items() if n >= CONFIRM_FRAMES}

        # Mașina de stări (din starea normală)
        if self.state == "normal":
            if "stop" in confirmed:
                self._go("slowing_stop", "stop")
                self._after(0.8, self._full_stop)      # 0.8 s încetinire → stop
            elif "yield" in confirmed:
                self._go("slowing_yield", "yield")
                self._after(1.5, self.resume)          # 1.5 s cedează → reia

        return min(self.multiplier, self._speed_cap), self.sign

    def _full_stop(self):
        self._go("stopped_sign", "stop")
        self._after(2.5, self.resume)                  # oprire 2.5 s → reia

    def resume(self):
        if self._timer: self._timer.cancel()
        self._go("normal", None); self._counts.clear()
```

---

## Anexa I — Detecția obstacolelor (`obstacle_detection.py`)

Abordarea „obiect pe drum": găsește asfaltul (regiunea întunecată), apoi caută
pete luminoase compacte DOAR pe drum, excluzând banda și semnele.

```python
def detect(self, frame, exclude_boxes=None):
    """Returnează {detected, bbox, area_frac, distance_frac, dangerous}."""
    h, w = frame.shape[:2]
    y0, y1 = int(h*ROI_TOP_FRAC), int(h*ROI_BOT_FRAC)
    x0, x1 = int(w*ROI_LEFT_FRAC), int(w*ROI_RIGHT_FRAC)
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]; roi_area = rh * rw
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV); v = hsv[:, :, 2]

    # 1. Drumul = cea mai mare regiune întunecată (asfaltul negru)
    dark = cv2.morphologyEx(cv2.inRange(v, 0, ROAD_V_MAX), cv2.MORPH_CLOSE, _K7, 2)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return _empty()
    road_c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(road_c) < MIN_ROAD_FRAC * roi_area:   # niciun drum clar
        return _empty()
    road_mask = np.zeros((rh, rw), np.uint8)
    cv2.drawContours(road_mask, [road_c], -1, 255, cv2.FILLED)

    # 2. Obiecte = pete luminoase PE drum, fără culori de bandă/SEMN
    yellow = cv2.inRange(hsv, _YELLOW_LO, _YELLOW_HI)
    red    = cv2.bitwise_or(cv2.inRange(hsv, _RED1_LO, _RED1_HI),
                            cv2.inRange(hsv, _RED2_LO, _RED2_HI))
    blue   = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
    sign_col = yellow | red | blue
    bright = cv2.inRange(v, ROAD_V_MAX + OBJ_V_MARGIN, 255)
    obj = cv2.bitwise_and(road_mask, bright)
    obj = cv2.bitwise_and(obj, cv2.bitwise_not(sign_col))
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN,  _K3, 1)
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, _K5, 1)

    excl = _inflate_boxes(exclude_boxes, SIGN_BOX_PAD) if exclude_boxes else []

    # 3. Cel mai mare blob COMPACT valid (dimensiune + solidity + raport laturi)
    best_area, best_bbox = 0, None
    for c in cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX: continue
        x, y, wc, hc = cv2.boundingRect(c)
        if not (MIN_ASPECT < wc / max(hc, 1) < MAX_ASPECT): continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if (area / hull if hull > 0 else 0) < MIN_SOLIDITY: continue
        full_box = (x0+x, y0+y, x0+x+wc, y0+y+hc)
        if any(_overlaps(full_box, e) for e in excl):       # exclude semnele
            continue
        if area > best_area:
            best_area, best_bbox = area, (x, y, wc, hc)

    if best_bbox is None: return _empty()
    area_frac = best_area / roi_area
    self._confirm = min(self._confirm + 1, CONFIRM_FRAMES * 4)
    if area_frac < DETECT_THRESHOLD or self._confirm < CONFIRM_FRAMES:
        return _empty(area_frac=round(area_frac, 4))
    rx, ry, rw_, rh_ = best_bbox
    return {"detected": True, "bbox": (x0+rx, y0+ry, rw_, rh_),
            "area_frac": round(area_frac, 4),
            "dangerous": area_frac >= DANGER_THRESHOLD,
            "distance_frac": round(min(1.0, area_frac / DANGER_THRESHOLD), 3)}
```

---

## Anexa J — Parcare automată nose-in (`parking_module.py`)

Mașină de stări simplă: rotire 180° pe loc (bazată pe timp) → intrare dreaptă în
garaj. Senzorii defecți sunt ignorați; citirile sunt filtrate cu median.

```python
class ParkingController:
    FRONT_SENSORS    = ("front_left", "front_center", "front_right")
    DISABLED_SENSORS = ("rear_center", "front_center")   # senzori arși → ignorați
    SENSOR_MEDIAN_N  = 5

    TURN_RATE   = 0.80;  TURN_DIR = +1;  TURN_TIME_S = 1.7   # ~180° (calibrabil)
    FORWARD_SPEED = 0.42; CREEP_SPEED = 0.36
    SLOW_CM = 35.0;  STOP_CM = 25.0;  WALL_RANGE_CM = 80.0
    LOOP_HZ = 12;  MAX_TIME_S = 30.0;  LOST_TIMEOUT_S = 9.0

    def _run(self):
        t0 = time.time(); dt = 1.0 / self.LOOP_HZ
        try:
            if not self._phase_turn(t0, dt): return
            self._phase_forward(t0, dt)
        except Exception as exc:
            self._finish("aborted", f"Eroare: {exc}")

    def _phase_turn(self, t0, dt):
        """Rotire 180° pe loc, BAZATĂ PE TIMP (fiabil, fără senzori)."""
        self._set_state("turn", f"Mă întorc 180°… ({self.TURN_TIME_S:.1f}s)")
        t_turn = time.time()
        while not self._stop_evt.is_set():
            time.sleep(dt)
            if self._timed_out(t0): return False
            self._read()
            if time.time() - t_turn >= self.TURN_TIME_S:
                self._drive(0.0, 0.0); return True
            self._drive(0.0, self.TURN_DIR * self.TURN_RATE)   # spin pe loc
        self._finish("aborted", "Oprit manual"); return False

    def _phase_forward(self, t0, dt):
        """Merge DREPT înainte (turn=0) până oricare senzor față ≤ STOP_CM."""
        self._set_state("forward", "Intru drept în garaj…")
        last_seen = time.time()
        while not self._stop_evt.is_set():
            time.sleep(dt)
            if self._timed_out(t0): return
            d = self._read()
            front = self._min_valid(d, self.FRONT_SENSORS)     # cel mai apropiat
            if front is not None and front < self.WALL_RANGE_CM:
                last_seen = time.time()
            if front is not None and front <= self.STOP_CM:
                self._finish("done", f"Parcat ✓ ({front:.0f} cm)"); return
            if time.time() - last_seen > self.LOST_TIMEOUT_S:
                self._finish("aborted", "Nu văd garajul — anulez"); return
            speed = self.CREEP_SPEED if (front and front < self.SLOW_CM) else self.FORWARD_SPEED
            self._drive(+speed, 0.0)                            # DREPT, fără viraje
        self._finish("aborted", "Oprit manual")

    def _read(self):
        """Citire + filtru MEDIAN per senzor; senzorii defecți scoși la sursă."""
        raw = self.get_distances() or {}
        for s in self.DISABLED_SENSORS:
            raw.pop(s, None)
        d = {}
        for name, val in raw.items():
            hist = self._hist.setdefault(name, [])
            if val is not None:
                hist.append(float(val))
                if len(hist) > self.SENSOR_MEDIAN_N: hist.pop(0)
            d[name] = sorted(hist)[len(hist) // 2] if hist else None
        with self._lock:
            self._last_dist = dict(d)
        return d

    @staticmethod
    def _min_valid(d, keys):
        vals = [d[k] for k in keys if d.get(k) is not None]
        return min(vals) if vals else None
```

---

## Anexa K — Funcții auxiliare de viziune (`utlis.py`)

Segmentarea drumului (foaie neagră între linii galbene), transformarea bird's-eye
și histograma de coloane folosite în calibrare/diagnostic.

```python
def thresholding_robust(img):
    """Detectare drum (foi negre/gri) pe imaginea warpată; exclude alb + galben."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask_valid  = cv2.threshold(v, 20,  255, cv2.THRESH_BINARY)[1]      # exclude bord warp
    mask_dark   = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY_INV)[1]  # exclude alb/lumină
    mask_yellow = cv2.inRange(hsv, np.array([10, 100, 80]), np.array([45, 255, 255]))
    mask = cv2.bitwise_and(mask_valid, mask_dark)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_yellow))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Cel mai mare blob = suprafața drumului
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1: return np.zeros_like(mask)
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    if stats[best, cv2.CC_STAT_AREA] < 500: return np.zeros_like(mask)
    return np.where(labels == best, np.uint8(255), np.uint8(0))


def warpImg(img, points, w, h, inv=False):
    """Transformare de perspectivă (bird's-eye view) sau inversa ei."""
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts2, pts1) if inv \
        else cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))


def getHistogram(img, minPer=0.1, region=1):
    """Histogramă de coloane → X-ul dominant (baza unei linii de bandă)."""
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0] // region:, :], axis=0)
    minValue = minPer * np.max(histValues)
    idx = np.where(histValues >= minValue)
    return int(np.average(idx)) if idx[0].size > 0 else img.shape[1] // 2
```

---

*Notă: codul complet (≈2700 linii în `app.py` + module) este disponibil în
arhiva proiectului. Anexele de mai sus conțin algoritmii esențiali din fiecare
componentă.*
