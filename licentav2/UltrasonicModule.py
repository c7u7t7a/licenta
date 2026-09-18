"""
UltrasonicModule.py — 6× HC-SR04 pe Raspberry Pi 4

Dispoziție fizică:
    ┌─────────────────────────────┐
    │  [F-St]   [F-C]   [F-Dr]   │  ← FAȚĂ  (3 senzori)
    │          MAȘINĂ             │
    │  [S-St]   [S-C]   [S-Dr]   │  ← SPATE (3 senzori)
    └─────────────────────────────┘

Pini (BCM) — senzori 3.3V compatibili, ECHO direct la GPIO (fără voltage divider):
    FAȚĂ-STÂNGA   TRIG=4,  ECHO=24
    FAȚĂ-CENTRU   TRIG=12, ECHO=25
    FAȚĂ-DREAPTA  TRIG=16, ECHO=26
    SPATE-STÂNGA  TRIG=20, ECHO=21
    SPATE-CENTRU  TRIG=18, ECHO=8
    SPATE-DREAPTA TRIG=7,  ECHO=9

Cablaj (HC-SR04+ / 3.3V-compatibil):
    VCC  → 3.3V Pi (Pin 1 sau 17)
    GND  → GND Pi
    TRIG → GPIO direct
    ECHO → GPIO direct (fără rezistențe!)
"""

from gpiozero import DistanceSensor, Device
from gpiozero.pins.rpigpio import RPiGPIOFactory
import threading

Device.pin_factory = RPiGPIOFactory()

# ── Configurare pini ────────────────────────────────────────────────────────
SENSOR_PINS = {
    # ── Față (3 senzori) ────────────────────────────────────────────────────
    "front_left":   {"trig": 4,  "echo": 24},
    "front_center": {"trig": 12, "echo": 25},
    "front_right":  {"trig": 16, "echo": 26},
    # ── Spate (3 senzori) ───────────────────────────────────────────────────
    "rear_left":    {"trig": 20, "echo": 21},
    "rear_center":  {"trig": 18, "echo": 8},
    "rear_right":   {"trig": 7,  "echo": 9},
}

MAX_DISTANCE_M = 3.0   # distanță maximă măsurată (m)


class UltrasonicArray:
    """
    Gestionează 6 senzori HC-SR04 simultan.

    Fiecare senzor rulează independent via gpiozero (non-blocking).
    Citirile sunt mereu disponibile via get_distances() sau get_closest().
    """

    def __init__(self):
        self._sensors: dict[str, DistanceSensor] = {}
        self._lock = threading.Lock()

        for name, pins in SENSOR_PINS.items():
            try:
                s = DistanceSensor(
                    echo=pins["echo"],
                    trigger=pins["trig"],
                    max_distance=MAX_DISTANCE_M,
                    queue_len=3,        # medie pe 3 citiri → stabil dar rapid
                    partial=True,       # CRITIC: .distance NU mai blochează dacă
                                        # un senzor nu primește echo (cablaj/GND).
                                        # Fără asta, firul de parcare îngheață.
                )
                self._sensors[name] = s
                print(f"[Ultrasonic] ✓ {name:14s} TRIG={pins['trig']:2d}  ECHO={pins['echo']:2d}")
            except Exception as e:
                print(f"[Ultrasonic] ✗ {name:14s} TRIG={pins['trig']:2d} ECHO={pins['echo']:2d} — {e}")

        ok = len(self._sensors)
        total = len(SENSOR_PINS)
        print(f"[Ultrasonic] {ok}/{total} senzori inițializați")
        if ok == 0:
            # Niciun senzor → semnalează eșec clar (app.py va marca parcarea
            # ca indisponibilă, în loc să arate o grilă tăcută de liniuțe).
            raise RuntimeError(
                "niciun senzor HC-SR04 inițializat — verifică alimentarea, "
                "cablajul și dacă pinii nu sunt ocupați (sudo killall python3)")

    @property
    def count(self) -> int:
        """Numărul de senzori care au pornit cu succes."""
        return len(self._sensors)

    # ── API public ────────────────────────────────────────────────────────────

    def get_distances(self) -> dict[str, float | None]:
        """
        Returnează distanțele curente în cm pentru toți senzorii.
        None dacă senzorul nu e disponibil sau citirea e eronată.
        """
        result = {}
        for name, sensor in self._sensors.items():
            try:
                result[name] = round(sensor.distance * 100, 1)
            except Exception:
                result[name] = None
        return result

    def get_distance(self, position: str) -> float | None:
        """
        Returnează distanța în cm pentru un singur senzor.
        position: 'front' | 'front_left' | 'front_right' | 'left' | 'right' | 'rear'
        """
        sensor = self._sensors.get(position)
        if sensor is None:
            return None
        try:
            return round(sensor.distance * 100, 1)
        except Exception:
            return None

    def get_closest(self) -> tuple[str, float] | tuple[None, None]:
        """
        Returnează (pozitie, distanta_cm) pentru cel mai aproape obiect.
        """
        distances = {k: v for k, v in self.get_distances().items() if v is not None}
        if not distances:
            return None, None
        closest = min(distances, key=distances.get)
        return closest, distances[closest]

    def any_closer_than(self, threshold_cm: float) -> bool:
        """True dacă orice senzor detectează un obiect mai aproape de threshold_cm."""
        return any(
            v is not None and v < threshold_cm
            for v in self.get_distances().values()
        )

    def front_distances(self) -> dict[str, float | None]:
        """Distanțele celor 3 senzori față {front_left, front_center, front_right}."""
        return {k: self.get_distance(k) for k in ("front_left", "front_center", "front_right")}

    def rear_distances(self) -> dict[str, float | None]:
        """Distanțele celor 3 senzori spate {rear_left, rear_center, rear_right}."""
        return {k: self.get_distance(k) for k in ("rear_left", "rear_center", "rear_right")}

    def front_center(self) -> float | None:
        """Scurtătură: senzorul central față (cel mai important pentru oprire)."""
        return self.get_distance("front_center")

    def rear_center(self) -> float | None:
        """Scurtătură: senzorul central spate (important la mers înapoi)."""
        return self.get_distance("rear_center")

    def any_front_closer_than(self, threshold_cm: float) -> bool:
        """True dacă oricare senzor față detectează obstacol mai aproape de threshold_cm."""
        return any(
            v is not None and v < threshold_cm
            for v in self.front_distances().values()
        )

    def any_rear_closer_than(self, threshold_cm: float) -> bool:
        """True dacă oricare senzor spate detectează obstacol mai aproape de threshold_cm."""
        return any(
            v is not None and v < threshold_cm
            for v in self.rear_distances().values()
        )

    def close(self):
        """Eliberează toți senzorii (apelat la shutdown)."""
        for s in self._sensors.values():
            try:
                s.close()
            except Exception:
                pass
        print("[Ultrasonic] Toți senzorii închiși.")


# ── Singleton — folosit din app.py ──────────────────────────────────────────
_array: UltrasonicArray | None = None


def get_array() -> UltrasonicArray:
    """Returnează instanța singleton a array-ului de senzori."""
    global _array
    if _array is None:
        _array = UltrasonicArray()
    return _array


# ── Compatibilitate cu codul vechi (un singur senzor) ────────────────────────
class UltrasonicSensor:
    """Wrapper de compatibilitate — citește senzorul față."""
    def __init__(self, trigger: int = 4, echo: int = 24,
                 max_distance: float = MAX_DISTANCE_M):
        self._sensor = DistanceSensor(echo=echo, trigger=trigger,
                                      max_distance=max_distance)

    def get_distance_cm(self) -> float | None:
        try:
            return round(self._sensor.distance * 100, 1)
        except Exception:
            return None
