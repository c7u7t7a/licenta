#!/usr/bin/env python3
"""
test_one_sensor.py — Test BRUT (raw RPi.GPIO) pentru UN singur HC-SR04.
Ocolește gpiozero complet → spune clar dacă problema e cablaj sau software.

Rulează pe Pi cu serverul OPRIT:
    sudo systemctl stop robot
    cd ~/Desktop/licentav2
    python3 test_one_sensor.py front_center      # sau alt nume de senzor
    sudo systemctl start robot

Nume valide: front_left front_center front_right rear_left rear_center rear_right
(Implicit: front_center)

Interpretare:
    - "ECHO nu a urcat niciodată" la TOATE pulsurile → senzorul nu pinguie /
      ECHO nelegat / TRIG nelegat / VCC lipsă pe ACEL senzor. Schimbă TRIG↔ECHO
      sau verifică firele acelui senzor.
    - Distanțe care scad când pui mâna în față → senzorul + cablajul sunt OK,
      iar problema era doar în gpiozero/timing (vezi nota pigpio la final).
"""

import sys
import time

PINS = {
    "front_left":   {"trig": 4,  "echo": 24},
    "front_center": {"trig": 12, "echo": 25},
    "front_right":  {"trig": 16, "echo": 26},
    "rear_left":    {"trig": 20, "echo": 21},
    "rear_center":  {"trig": 18, "echo": 8},
    "rear_right":   {"trig": 7,  "echo": 9},
}

name = sys.argv[1] if len(sys.argv) > 1 else "front_center"
if name not in PINS:
    print("Nume invalid. Alege:", ", ".join(PINS)); sys.exit(1)

TRIG = PINS[name]["trig"]
ECHO = PINS[name]["echo"]

import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)
time.sleep(0.3)

print(f"Testez '{name}'  TRIG={TRIG}  ECHO={ECHO}  (Ctrl+C pt stop)\n")

def ping():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)          # 10 µs
    GPIO.output(TRIG, False)
    t_limit = time.time() + 0.04
    # așteaptă ECHO HIGH
    while GPIO.input(ECHO) == 0:
        if time.time() > t_limit:
            return None          # ECHO nu a urcat → fără echo
    t_start = time.time()
    # așteaptă ECHO LOW
    while GPIO.input(ECHO) == 1:
        if time.time() > t_limit:
            return None
    t_end = time.time()
    return (t_end - t_start) * 34300.0 / 2.0   # cm

try:
    while True:
        d = ping()
        if d is None:
            print("  ECHO nu a urcat niciodată  → fără echo (cablaj/VCC/TRIG-ECHO)")
        else:
            print(f"  {d:6.1f} cm")
        time.sleep(0.3)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
    print("\nGata.")
