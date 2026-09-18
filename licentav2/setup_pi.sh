#!/bin/bash
# ============================================================
#  setup_pi.sh — Configurare completă Raspberry Pi 4
#  Raspberry Pi OS (Bookworm / Bullseye 64-bit), fresh install
#
#  Rulare:
#      chmod +x setup_pi.sh
#      ./setup_pi.sh
# ============================================================

set -e   # oprește la prima eroare

echo ""
echo "========================================================"
echo "  Autonomous Car — Raspberry Pi Setup"
echo "========================================================"
echo ""

# ── 1. Update sistem ─────────────────────────────────────────
echo "[1/7] Update pachete sistem..."
sudo apt update && sudo apt upgrade -y

# ── 2. Dependențe sistem ─────────────────────────────────────
echo "[2/7] Instalare dependențe sistem..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-numpy \
    libatlas-base-dev \
    libjpeg-dev \
    libopenjp2-7 \
    libsdl2-dev \
    libsdl2-mixer-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-ttf-2.0-0 \
    python3-pygame \
    git \
    v4l-utils \
    libcamera-apps

# ── 3. Virtual environment ───────────────────────────────────
echo "[3/7] Creare virtual environment..."
cd ~
python3 -m venv licenta_env --system-site-packages
# --system-site-packages permite accesul la python3-opencv și python3-pygame din apt
source licenta_env/bin/activate

# ── 4. Pip packages ──────────────────────────────────────────
echo "[4/7] Instalare pachete Python..."
pip install --upgrade pip
pip install -r ~/licenta/requirements_pi.txt

# ── 5. Test GPIO ─────────────────────────────────────────────
echo "[5/7] Verificare GPIO..."
python3 -c "from gpiozero import Device; print('  gpiozero OK')" || \
    echo "  WARNING: gpiozero nu funcționează — verifică rpi-lgpio"

# ── 6. Test Camera ───────────────────────────────────────────
echo "[6/7] Verificare cameră..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ok = cap.isOpened()
cap.release()
print('  Camera index 0:', 'OK' if ok else 'NU GĂSITĂ (normal dacă nu e conectată)')
"

# ── 7. Test OpenCV ───────────────────────────────────────────
echo "[7/7] Verificare OpenCV + DNN..."
python3 -c "
import cv2, numpy
print(f'  OpenCV: {cv2.__version__}')
print(f'  NumPy:  {numpy.__version__}')
import os
model = os.path.expanduser('~/licenta/traffic_sign_classifier.onnx')
if os.path.exists(model):
    net = cv2.dnn.readNetFromONNX(model)
    print(f'  ONNX model: OK')
else:
    print(f'  ONNX model: LIPSĂ — copiaza traffic_sign_classifier.onnx pe Pi')
"

echo ""
echo "========================================================"
echo "  SETUP COMPLET!"
echo ""
echo "  Activare environment:"
echo "    source ~/licenta_env/bin/activate"
echo ""
echo "  Pornire server:"
echo "    cd ~/licenta"
echo "    python app.py"
echo ""
echo "  Acces browser:"
echo "    http://$(hostname -I | awk '{print $1}'):5000"
echo "========================================================"
