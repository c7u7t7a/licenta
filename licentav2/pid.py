"""
pid.py — Thread-safe PID controller for lane-keeping.

    error  = curve value from lane detection ∈ [-1, 1]
    output = corrected turn command ∈ [-1, 1]

Tuning guide (Ziegler-Nichols lite):
    1. Start Kp=0.5, Ki=0.0, Kd=0.0 — car follows lane roughly.
    2. Raise Kp until the car oscillates left-right → that's Ku.
    3. Set Kp=0.6*Ku, Ki=1.2*Ku/Tu, Kd=0.075*Ku*Tu  (Tu = oscillation period).
    4. Good starting point: Kp=0.7, Ki=0.02, Kd=0.15

Features:
    - Anti-windup integrator clamp (prevents integral wind-up in saturation)
    - Low-pass filter on derivative (α=0.8) to suppress sensor noise
    - Thread-safe via internal Lock
    - Diagnostic state dict for real-time UI
"""

import time
import threading


class PIDController:
    """
    Discrete-time PID with derivative filtering and anti-windup.

    Usage:
        pid = PIDController(kp=0.7, ki=0.02, kd=0.15)
        ...
        # in lane-assist loop:
        turn = pid.compute(curve)
    """

    def __init__(self, kp: float = 0.7, ki: float = 0.02, kd: float = 0.15):
        self._lock = threading.Lock()

        # Gains
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        # Internal state
        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0   # EMA-filtered derivative
        self._prev_time  = None  # monotonic time of last call

        # Diagnostics (updated every compute() call)
        self.last_p      = 0.0
        self.last_i      = 0.0
        self.last_d      = 0.0
        self.last_output = 0.0
        self.last_error  = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, error: float) -> float:
        """
        Call once per lane-detect cycle.

        Args:
            error: signed lane deviation ∈ [-1, 1]  (positive = drift right)

        Returns:
            turn command ∈ [-1, 1]
        """
        now = time.monotonic()
        with self._lock:
            # dt — clamped to realistic range (5 ms … 200 ms)
            if self._prev_time is None:
                dt = 0.05
            else:
                dt = now - self._prev_time
                dt = max(0.005, min(dt, 0.20))
            self._prev_time = now

            # ── Proportional ──────────────────────────────────────────────────
            p = self.kp * error

            # ── Integral with anti-windup clamp ───────────────────────────────
            self._integral += error * dt
            self._integral  = max(-1.5, min(1.5, self._integral))
            i = self.ki * self._integral

            # ── Derivative with EMA low-pass (α = 0.9) ────────────────────────
            # α mai mare = filtru mai agresiv = mai puțin zgomot la curbe
            raw_d = (error - self._prev_error) / dt
            self._d_filtered = 0.9 * self._d_filtered + 0.1 * raw_d
            d = self.kd * self._d_filtered

            self._prev_error = error

            # ── Sum + clamp ────────────────────────────────────────────────────
            output = max(-1.0, min(1.0, p + i + d))

            # Save diagnostics
            self.last_error  = round(error,  4)
            self.last_p      = round(p,      4)
            self.last_i      = round(i,      4)
            self.last_d      = round(d,      4)
            self.last_output = round(output, 4)

        return output

    def reset(self):
        """Zero integrator and derivative state. Call on mode switches."""
        with self._lock:
            self._integral   = 0.0
            self._prev_error = 0.0
            self._d_filtered = 0.0
            self._prev_time  = None
            self.last_p = self.last_i = self.last_d = self.last_output = self.last_error = 0.0

    def set_gains(self,
                  kp: float = None,
                  ki: float = None,
                  kd: float = None):
        """Update gains without restarting. Thread-safe."""
        with self._lock:
            if kp is not None: self.kp = max(0.0, float(kp))
            if ki is not None: self.ki = max(0.0, float(ki))
            if kd is not None: self.kd = max(0.0, float(kd))

    def get_state(self) -> dict:
        """Return current gains + last-computed diagnostics for the UI."""
        with self._lock:
            return {
                "kp":       round(self.kp, 3),
                "ki":       round(self.ki, 3),
                "kd":       round(self.kd, 3),
                "error":    self.last_error,
                "P":        self.last_p,
                "I":        self.last_i,
                "D":        self.last_d,
                "output":   self.last_output,
                "integral": round(self._integral, 4),
            }
