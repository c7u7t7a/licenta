"""
DS4Module.py — DualShock 4 / PS4 Controller integration
=========================================================
Rulează ca thread daemon în app.py.  Detectează automat controllerul,
suportă hotplug (reconectare fără restart), și expune starea curentă
prin proprietăți thread-safe.

Mapare butoane (Linux/Pi + Windows):
─────────────────────────────────────
  Stânga stick Y  → viteză (forward / backward)
  Stânga stick X  → viraj stânga / dreapta
  Dreapta stick X → viraj alternativ (pentru manevre fine)
  L2 analog       → turbo — multiplică viteza cu 1.0 (eliberează limita)
  R2 analog       → frână — reduce viteza la 40%

  ✕  (Cross)     → Emergency Stop
  ○  (Circle)    → Toggle Înregistrare
  □  (Square)    → Toggle Detecție Semne
  △  (Triangle)  → Toggle Lane Assist / Manual
  L1             → Reduce viteză de bază (−5%)
  R1             → Crește  viteză de bază (+5%)
  Share          → Toggle Mock Lane Mode
  Options        → Toggle Detecție Obstacole
  D-pad Up/Down  → Ajustează viteză de bază ±5%
  PS Button      → (ignorat — pentru siguranță)

Condiții:
  • pip install pygame  (sau pygame-ce pentru Pi)
  • DS4 conectat USB sau Bluetooth înainte de pornire
    (hotplug detectat automat la fiecare 2 secunde)
"""

import os
import threading
import time

# Headless display — obligatoriu pe Pi fără monitor
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ── Constante de mapare ────────────────────────────────────────────────────────

# Axe (index pygame)
AXIS_LS_X  = 0   # Left Stick stânga/dreapta
AXIS_LS_Y  = 1   # Left Stick sus/jos  (negativ = sus = forward)
AXIS_L2    = 2   # L2 trigger  -1.0 (eliberat) … +1.0 (apăsat)
AXIS_RS_X  = 3   # Right Stick stânga/dreapta
AXIS_RS_Y  = 4   # Right Stick sus/jos
AXIS_R2    = 5   # R2 trigger  -1.0 … +1.0

# Butoane (index pygame — standard DS4 pe Linux/Pi; Windows similar)
BTN_CROSS     = 0
BTN_CIRCLE    = 1
BTN_SQUARE    = 2
BTN_TRIANGLE  = 3
BTN_L1        = 4
BTN_R1        = 5
BTN_L2_DIG    = 6   # L2 digital (click complet)
BTN_R2_DIG    = 7
BTN_SHARE     = 8
BTN_OPTIONS   = 9
BTN_L3        = 10  # Left Stick apăsat
BTN_R3        = 11
BTN_PS        = 12  # PS / Home
BTN_TOUCHPAD  = 13

DEADZONE = 0.08    # ignoră mișcări mici ale stick-urilor (drift)
POLL_HZ  = 50      # frecvența de citire controller (50 Hz)


class DS4Controller:
    """
    Thread-safe wrapper în jurul pygame.joystick.

    Utilizare:
        ds4 = DS4Controller()
        ds4.start()                  # pornește thread-ul
        speed, turn = ds4.get_axes() # citește în orice moment
        ds4.rumble(0.5, 0.5, 300)    # feedback haptic (milisecunde)
    """

    def __init__(self):
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None

        # State
        self._connected = False
        self._joy       = None
        self._axes      = [0.0] * 6
        self._buttons   = [0]   * 16
        self._hat       = (0, 0)

        # Callbacks setate din app.py
        self.on_cross     = None   # emergency stop
        self.on_circle    = None   # toggle recording
        self.on_square    = None   # toggle signs
        self.on_triangle  = None   # toggle mode
        self.on_l1        = None   # speed down
        self.on_r1        = None   # speed up
        self.on_share     = None   # toggle mock lane
        self.on_options   = None   # toggle obstacles
        self.on_hat       = None   # dpad: (dx, dy)

        # Pressed-once tracking (evită repetiții)
        self._prev_buttons = [0] * 16
        self._prev_hat     = (0, 0)

    # ── Proprietăți publice ────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def get_axes(self) -> tuple:
        """Returnează (speed, turn) aplicate deadzoneului, ∈ [-1,1]."""
        with self._lock:
            ls_y = self._axes[AXIS_LS_Y] if len(self._axes) > AXIS_LS_Y else 0.0
            ls_x = self._axes[AXIS_LS_X] if len(self._axes) > AXIS_LS_X else 0.0
            rs_x = self._axes[AXIS_RS_X] if len(self._axes) > AXIS_RS_X else 0.0
            l2   = self._axes[AXIS_L2]   if len(self._axes) > AXIS_L2   else -1.0
            r2   = self._axes[AXIS_R2]   if len(self._axes) > AXIS_R2   else -1.0

        # Trigger: valoarea normalizată 0…1
        l2_norm = (l2 + 1.0) / 2.0   # -1=neapăsat → 0,  +1=apăsat → 1
        r2_norm = (r2 + 1.0) / 2.0

        # Viteză: stânga stick Y (inversat), modificată de trigger-e
        speed = -_dz(ls_y)

        # Turbo cu L2: eliberează limita de viteză (1.0 în loc de BASE_SPEED)
        # Frână cu R2: reduce viteza
        if l2_norm > 0.1:
            speed = speed * (1.0 + l2_norm * 0.35)  # +35% max turbo
        if r2_norm > 0.1:
            speed = speed * max(0.0, 1.0 - r2_norm * 0.6)

        speed = max(-1.0, min(1.0, speed))

        # Viraj: left stick X + right stick X (ponderate)
        turn = _dz(ls_x) * 0.7 + _dz(rs_x) * 0.3
        turn = max(-1.0, min(1.0, turn))

        return speed, turn

    def get_state_dict(self) -> dict:
        """Starea completă pentru broadcast Socket.IO."""
        with self._lock:
            axes    = list(self._axes)
            buttons = list(self._buttons)
            hat     = self._hat
            conn    = self._connected

        speed, turn = self.get_axes()
        return {
            "connected": conn,
            "speed":     round(speed, 3),
            "turn":      round(turn,  3),
            "axes": {
                "ls_x":  round(axes[AXIS_LS_X] if len(axes) > AXIS_LS_X else 0, 3),
                "ls_y":  round(axes[AXIS_LS_Y] if len(axes) > AXIS_LS_Y else 0, 3),
                "rs_x":  round(axes[AXIS_RS_X] if len(axes) > AXIS_RS_X else 0, 3),
                "rs_y":  round(axes[AXIS_RS_Y] if len(axes) > AXIS_RS_Y else 0, 3),
                "l2":    round((axes[AXIS_L2] + 1) / 2 if len(axes) > AXIS_L2 else 0, 3),
                "r2":    round((axes[AXIS_R2] + 1) / 2 if len(axes) > AXIS_R2 else 0, 3),
            },
            "buttons": {
                "cross":    buttons[BTN_CROSS]    if len(buttons) > BTN_CROSS    else 0,
                "circle":   buttons[BTN_CIRCLE]   if len(buttons) > BTN_CIRCLE   else 0,
                "square":   buttons[BTN_SQUARE]   if len(buttons) > BTN_SQUARE   else 0,
                "triangle": buttons[BTN_TRIANGLE] if len(buttons) > BTN_TRIANGLE else 0,
                "l1":       buttons[BTN_L1]       if len(buttons) > BTN_L1       else 0,
                "r1":       buttons[BTN_R1]       if len(buttons) > BTN_R1       else 0,
                "share":    buttons[BTN_SHARE]    if len(buttons) > BTN_SHARE    else 0,
                "options":  buttons[BTN_OPTIONS]  if len(buttons) > BTN_OPTIONS  else 0,
                "l3":       buttons[BTN_L3]       if len(buttons) > BTN_L3       else 0,
                "r3":       buttons[BTN_R3]       if len(buttons) > BTN_R3       else 0,
                "ps":       buttons[BTN_PS]       if len(buttons) > BTN_PS       else 0,
            },
            "dpad": {"x": hat[0], "y": hat[1]},
        }

    def rumble(self, low: float = 0.5, high: float = 0.5, duration_ms: int = 200):
        """Feedback haptic — necesită pygame 2.0.6+ și driver DS4 cu rumble."""
        try:
            with self._lock:
                joy = self._joy
            if joy is not None:
                joy.rumble(low, high, duration_ms)
        except Exception:
            pass

    # ── Thread principal ───────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        try:
            import pygame
        except ImportError:
            print("[DS4] pygame nu este instalat — pip install pygame")
            return

        pygame.init()
        pygame.joystick.init()

        print("[DS4] thread pornit — aștept controller…")

        while self._running:
            # ── Hotplug: încearcă să conecteze dacă nu e conectat ──────────────
            with self._lock:
                conn = self._connected

            if not conn:
                if pygame.joystick.get_count() > 0:
                    try:
                        joy = pygame.joystick.Joystick(0)
                        joy.init()
                        with self._lock:
                            self._joy       = joy
                            self._connected = True
                            self._axes      = [0.0] * max(joy.get_numaxes(), 6)
                            self._buttons   = [0]   * max(joy.get_numbuttons(), 14)
                            self._hat       = (0, 0)
                        print(f"[DS4] conectat: {joy.get_name()}  "
                              f"({joy.get_numaxes()} axe, {joy.get_numbuttons()} butoane)")
                    except Exception as exc:
                        print(f"[DS4] eroare la inițializare: {exc}")
                        time.sleep(2.0)
                        continue
                else:
                    time.sleep(2.0)
                    continue

            # ── Poll evenimente ───────────────────────────────────────────────
            try:
                for event in pygame.event.get():
                    if event.type == pygame.JOYAXISMOTION:
                        with self._lock:
                            if event.axis < len(self._axes):
                                self._axes[event.axis] = round(event.value, 3)

                    elif event.type == pygame.JOYBUTTONDOWN:
                        with self._lock:
                            if event.button < len(self._buttons):
                                self._buttons[event.button] = 1

                    elif event.type == pygame.JOYBUTTONUP:
                        with self._lock:
                            if event.button < len(self._buttons):
                                self._buttons[event.button] = 0

                    elif event.type == pygame.JOYHATMOTION:
                        with self._lock:
                            self._hat = event.value

                    elif event.type in (pygame.JOYDEVICEADDED,):
                        pass   # se va reconecta în iterația următoare

                    elif event.type in (pygame.JOYDEVICEREMOVED,):
                        with self._lock:
                            self._connected = False
                            self._joy       = None
                        print("[DS4] controller deconectat")

            except Exception as exc:
                print(f"[DS4] eroare de citire: {exc}")
                with self._lock:
                    self._connected = False
                    self._joy       = None
                time.sleep(1.0)
                continue

            # ── Detectare apăsări noi (edge-triggered) ────────────────────────
            with self._lock:
                cur_btn = list(self._buttons)
                cur_hat = self._hat

            self._fire_callbacks(cur_btn, cur_hat)

            self._prev_buttons = list(cur_btn)
            self._prev_hat     = cur_hat

            time.sleep(1.0 / POLL_HZ)

    def _fire_callbacks(self, cur_btn: list, cur_hat: tuple):
        """Apelează callback-urile la rising edge (apăsare nouă)."""
        def _rose(idx):
            return (idx < len(cur_btn) and cur_btn[idx] == 1
                    and (idx >= len(self._prev_buttons) or self._prev_buttons[idx] == 0))

        if _rose(BTN_CROSS)    and self.on_cross:    self.on_cross()
        if _rose(BTN_CIRCLE)   and self.on_circle:   self.on_circle()
        if _rose(BTN_SQUARE)   and self.on_square:   self.on_square()
        if _rose(BTN_TRIANGLE) and self.on_triangle: self.on_triangle()
        if _rose(BTN_L1)       and self.on_l1:       self.on_l1()
        if _rose(BTN_R1)       and self.on_r1:       self.on_r1()
        if _rose(BTN_SHARE)    and self.on_share:    self.on_share()
        if _rose(BTN_OPTIONS)  and self.on_options:  self.on_options()

        # D-pad (hat): trimite callback la orice schimbare
        if cur_hat != self._prev_hat and self.on_hat:
            self.on_hat(cur_hat[0], cur_hat[1])


# ── Utilitar deadzone ──────────────────────────────────────────────────────────

def _dz(val: float, zone: float = DEADZONE) -> float:
    """Aplică deadzone liniar și remapează 0 la 1."""
    if abs(val) < zone:
        return 0.0
    sign = 1.0 if val > 0 else -1.0
    return sign * (abs(val) - zone) / (1.0 - zone)
