#!/usr/bin/env python3
"""
test_sensors.py — Test rapid pentru cei 6 senzori HC-SR04.

Rulează DIRECT pe Pi, cu serverul OPRIT (altfel pinii sunt ocupați):

    sudo systemctl stop robot
    cd ~/Desktop/licentav2
    python3 test_sensors.py
    # Ctrl+C pentru oprire, apoi:  sudo systemctl start robot

Ce arată:
    - la pornire: ce senzori s-au inițializat (✓/✗)
    - apoi, de 3×/secundă, distanța fiecărui senzor în cm
      "--" = None (citire eșuată)   |   ~300 = nimic în rază / echo deconectat

Cum interpretezi:
    - Dacă scriptul ÎNGHEAȚĂ și nu printează nimic → un ECHO e blocat
      (cablaj greșit / lipsă GND comun / senzor nealimentat).
    - Dacă un senzor arată mereu "--" sau ~300 indiferent de mână în față →
      acel senzor nu funcționează (verifică TRIG/ECHO/VCC/GND).
    - Mișcă mâna la 10–20 cm în fața fiecărui senzor și vezi dacă valoarea scade.
"""

import time
import sys
import os
import warnings

warnings.simplefilter("ignore")   # taie spam-ul "no echo received" ca să vezi grila

sys.path.insert(0, os.path.dirname(__file__))
from UltrasonicModule import get_array, SENSOR_PINS

ORDER = ["front_left", "front_center", "front_right",
         "rear_left",  "rear_center",  "rear_right"]


def main():
    print("Inițializez array-ul de senzori…\n")
    try:
        ar = get_array()
    except Exception as e:
        print(f"\n✗ Array-ul nu a putut porni: {e}")
        print("  Verifică alimentarea senzorilor și dacă serverul e oprit (sudo systemctl stop robot).")
        return

    print(f"\nCitesc {ar.count}/6 senzori. Mișcă mâna în fața fiecăruia. Ctrl+C pentru stop.\n")
    hdr = "  ".join(f"{n.replace('front','F').replace('rear','S'):>9}" for n in ORDER)
    print(hdr)
    print("-" * len(hdr))

    try:
        while True:
            d = ar.get_distances()
            cells = []
            for n in ORDER:
                v = d.get(n)
                cells.append(f"{v:6.0f}cm" if v is not None else f"{'--':>8}")
            print("  ".join(f"{c:>9}" for c in cells))
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nOpresc…")
    finally:
        try:
            ar.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
