"""
parking_module.py — Parcare automată cu FAȚA (nose-in), simplă: 180° + înainte.
==============================================================================

Scenariu (confirmat cu utilizatorul):
    Mașina pornește ÎNTOARSĂ INVERS (cu spatele spre garaj). Manevra are DOAR
    două mișcări:
        1. Se rotește 180° pe loc (acum FAȚA e spre garaj).
        2. Merge DREPT ÎNAINTE în garaj până atinge peretele din fund → STOP.
    FĂRĂ corecții de direcție / fără viraje („no swerving") — doar drept înainte.

Hardware:
    6× HC-SR04 (vezi UltrasonicModule.py). ⚠️ rear_center e ARS → ignorat.
    Pentru oprire se folosesc senzorii din FAȚĂ (front_left/center/right).
    Citirile sunt filtrate cu median → un senzor „flaky" nu strică oprirea.

Mașina e skid-steer (tank drive) → se rotește pe loc cu motor.move(0, ±turn).

Reglaje importante:
    TURN_TIME_S — durata rotirii de 180°. CALIBREAZĂ pe mașina ta.
    TURN_DIR    — sensul rotirii (+1 dreapta, -1 stânga).
    STOP_CM     — la ce distanță de perete se oprește.
"""

import threading
import time


class ParkingController:
    # ── Senzori ─────────────────────────────────────────────────────────────────
    FRONT_SENSORS    = ("front_left", "front_center", "front_right")
    # ⚠️ rear_center ARS + front_center citește random → ignorate amândouă.
    # Oprirea folosește „oricare din senzorii față rămași" = front_left / front_right.
    # Dacă de fapt alt senzor față e cel stricat, schimbă aici.
    DISABLED_SENSORS = ("rear_center", "front_center")
    SENSOR_MEDIAN_N  = 5                   # filtru median per senzor (anti-random)

    # ── Faza 1: TURN (rotire 180° pe loc, bazată pe timp) ───────────────────────
    TURN_RATE   = 0.80   # magnitudinea rotirii (motor.move(0, ±TURN_RATE))
    TURN_DIR    = +1     # +1 = rotește spre dreapta, -1 = spre stânga
    TURN_TIME_S = 1.7    # ⚠️ DURATA rotirii de ~180° — CALIBREAZĂ! crește dacă nu
                         #    face 180 complet, scade dacă trece de 180.

    # ── Faza 2: FORWARD (intră drept în garaj) ──────────────────────────────────
    FORWARD_SPEED = 0.42  # viteză de intrare
    CREEP_SPEED   = 0.36  # viteză redusă aproape de perete
    SLOW_CM       = 35.0  # sub atât → creep (peste STOP ca să nu treacă de 25)
    STOP_CM       = 25.0  # oprește când ORICARE senzor față e ≤ atât
    WALL_RANGE_CM = 80.0  # citire față sub atât = „vede garajul"

    # ── Generale ────────────────────────────────────────────────────────────────
    LOOP_HZ        = 12
    MAX_TIME_S     = 30.0  # timp total maxim pentru toată manevra
    LOST_TIMEOUT_S = 9.0   # în FORWARD: niciun perete văzut atâta timp → abort

    def __init__(self, motor, motor_lock, get_distances, on_status=None):
        """
        motor        : obiect cu .move(speed, turn) și .stop()
        motor_lock   : threading.Lock partajat cu restul aplicației
        get_distances: callable() -> dict {nume: cm|None} pentru cei 6 senzori
        on_status    : callback opțional(dict) apelat la fiecare update de stare
        """
        self.motor         = motor
        self.motor_lock    = motor_lock
        self.get_distances = get_distances
        self.on_status     = on_status

        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._thread    = None
        self._active    = False
        self._state     = "idle"     # idle|turn|forward|done|aborted
        self._message   = "Inactiv"
        self._last_dist = {}
        self._hist      = {}         # istoric per senzor pentru filtru median

    # ── API public ──────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def get_state(self) -> dict:
        with self._lock:
            return {
                "active":    self._active,
                "state":     self._state,
                "message":   self._message,
                "distances": dict(self._last_dist),
            }

    def start(self, turn_dir: int = None) -> bool:
        """Pornește manevra. Returnează False dacă deja rulează."""
        with self._lock:
            if self._active:
                return False
            self._active   = True
            self._state    = "starting"
            self._message  = "Pornesc parcarea…"
            self._stop_evt = threading.Event()
            self._hist.clear()
            if turn_dir in (+1, -1):
                self.TURN_DIR = turn_dir
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._emit()
        return True

    def stop(self):
        """Cere oprirea manevrei (motoarele se opresc în firul de execuție)."""
        self._stop_evt.set()

    # ── Bucla principală ─────────────────────────────────────────────────────────

    def _run(self):
        t0 = time.time()
        dt = 1.0 / self.LOOP_HZ
        try:
            if not self._phase_turn(t0, dt):
                return
            self._phase_forward(t0, dt)
        except Exception as exc:
            self._finish("aborted", f"Eroare: {exc}")

    # ── Faza 1: TURN (rotire 180° pe loc, BAZATĂ PE TIMP) ────────────────────────

    def _phase_turn(self, t0, dt) -> bool:
        """Se rotește pe loc pentru TURN_TIME_S → ~180°. După, FAȚA e spre garaj."""
        self._set_state("turn", f"Mă întorc 180°… ({self.TURN_TIME_S:.1f}s)")
        t_turn = time.time()
        while not self._stop_evt.is_set():
            time.sleep(dt)
            if self._timed_out(t0):
                return False
            self._read()   # actualizează grila de senzori în UI
            if time.time() - t_turn >= self.TURN_TIME_S:
                self._drive(0.0, 0.0)
                self._set_state("turn", "Întoarcere completă")
                return True
            self._drive(0.0, self.TURN_DIR * self.TURN_RATE)
        self._finish("aborted", "Oprit manual")
        return False

    # ── Faza 2: FORWARD (intră DREPT în garaj) ───────────────────────────────────

    def _phase_forward(self, t0, dt):
        """Merge DREPT înainte (fără viraje) până la peretele din fund → STOP."""
        self._set_state("forward", "Intru drept în garaj…")
        last_seen = time.time()
        while not self._stop_evt.is_set():
            time.sleep(dt)
            if self._timed_out(t0):
                return
            d = self._read()
            # Oprire când ORICARE senzor față e sub prag → MIN (cel mai apropiat).
            # Citirile sunt deja filtrate median per senzor (anti-random spikes).
            front = self._min_valid(d, self.FRONT_SENSORS)

            if front is not None and front < self.WALL_RANGE_CM:
                last_seen = time.time()

            # Am ajuns la perete (oricare senzor față ≤ STOP_CM)?
            if front is not None and front <= self.STOP_CM:
                self._finish("done", f"Parcat ✓  ({front:.0f} cm)")
                return

            # Nu mai văd garajul de prea mult timp → abort
            if time.time() - last_seen > self.LOST_TIMEOUT_S:
                self._finish("aborted", "Nu văd garajul în față — anulez")
                return

            speed = self.CREEP_SPEED if (front is not None and front < self.SLOW_CM) \
                else self.FORWARD_SPEED
            self._drive(+speed, 0.0)   # DREPT — turn=0, fără viraje
            self._set_state("forward", f"Față: {self._fmt(front)}")
        self._finish("aborted", "Oprit manual")

    # ── Utilitare ─────────────────────────────────────────────────────────────────

    def _read(self) -> dict:
        try:
            raw = self.get_distances() or {}
        except Exception:
            raw = {}
        for s in self.DISABLED_SENSORS:
            raw.pop(s, None)
        # Filtru MEDIAN per senzor → respinge valorile random (senzor flaky)
        d = {}
        for name, val in raw.items():
            hist = self._hist.setdefault(name, [])
            if val is not None:
                hist.append(float(val))
                if len(hist) > self.SENSOR_MEDIAN_N:
                    hist.pop(0)
            d[name] = sorted(hist)[len(hist) // 2] if hist else None
        with self._lock:
            self._last_dist = dict(d)
        return d

    @staticmethod
    def _min_valid(d: dict, keys) -> float | None:
        """Cel mai mic senzor valid → oprire când ORICARE e sub prag."""
        vals = [d[k] for k in keys if d.get(k) is not None]
        return min(vals) if vals else None

    def _timed_out(self, t0) -> bool:
        if time.time() - t0 > self.MAX_TIME_S:
            self._finish("aborted", "Timp expirat — anulez parcarea")
            return True
        return False

    def _drive(self, speed, turn):
        with self.motor_lock:
            self.motor.move(speed=speed, turn=turn)

    def _finish(self, state, msg):
        try:
            self._drive(0.0, 0.0)
        except Exception:
            pass
        with self._lock:
            self._active  = False
            self._state   = state
            self._message = msg
        self._emit()

    def _set_state(self, state, msg):
        with self._lock:
            self._state   = state
            self._message = msg
        self._emit()

    def _emit(self):
        if self.on_status:
            try:
                self.on_status(self.get_state())
            except Exception:
                pass

    @staticmethod
    def _fmt(v) -> str:
        return f"{v:.0f} cm" if v is not None else "—"
