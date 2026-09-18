"""
Behavioral Cloning — End-to-End Steering from Recorded Datasets.

Architecture: lightweight CNN regressor
  Input : 160×80 grayscale image (bottom crop of the camera frame)
  Output: scalar turn ∈ [-1, 1]
  Loss  : MSE on the `turn` column from the recording CSV

Usage (called from app.py via /train/start):
    from behavioral_cloning import train_model
    model_path = train_model(
        session_dirs=["/path/to/dataset/session_xxx"],
        epochs=20,
        lr=1e-3,
        progress_cb=lambda ep, total, loss, val_loss: ...
    )
    # model_path is an ONNX file ready for inference on the Pi

Inference (called from the lane-assist thread):
    from behavioral_cloning import BehavioralDriver
    driver = BehavioralDriver("models/steering_YYYYMMDD_HHMMSS.onnx")
    turn = driver.predict(bgr_frame)   # returns float ∈ [-1, 1]
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_W  = 160
IMG_H  = 80
# Fraction of the frame height to keep (bottom portion — where the road is)
CROP_TOP_FRAC = 0.40

_BASE       = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(_BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """Crop bottom, resize to IMG_W×IMG_H, convert to grayscale float32 [0,1]."""
    h = bgr.shape[0]
    crop_y = int(h * CROP_TOP_FRAC)
    roi    = bgr[crop_y:, :]
    small  = cv2.resize(roi, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _load_dataset(session_dirs: list[str]):
    """
    Load all (image, turn) pairs from the given session directories.
    Returns two numpy arrays: images (N, 1, H, W) and labels (N,).
    """
    images_list = []
    labels_list = []

    for sess_dir in session_dirs:
        img_dir  = os.path.join(sess_dir, "images")
        csv_path = os.path.join(sess_dir, "labels.csv")
        if not os.path.isdir(img_dir) or not os.path.exists(csv_path):
            print(f"[bc] skipping {sess_dir} — missing images/ or labels.csv")
            continue

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            fname = row.get("filename", "")
            turn  = float(row.get("turn", 0.0))
            fpath = os.path.join(img_dir, fname)
            if not os.path.exists(fpath):
                continue
            bgr = cv2.imread(fpath)
            if bgr is None:
                continue
            img = _preprocess(bgr)
            images_list.append(img[np.newaxis, :, :])   # (1, H, W)
            labels_list.append(turn)

    if not images_list:
        raise ValueError("Nicio imagine validă găsită în sesiunile selectate.")

    X = np.stack(images_list, axis=0)  # (N, 1, H, W)
    y = np.array(labels_list, dtype=np.float32)
    print(f"[bc] dataset: {len(X)} samples loaded")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def _build_model():
    """
    Build a small PyTorch CNN regressor.
    Returns the model (on CPU).
    Raises ImportError if PyTorch is not installed.
    """
    import torch
    import torch.nn as nn

    class SteeringCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # 80×40
                nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 40×20
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 20×10
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 10×5
                nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 10 * 5, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Tanh(),   # output ∈ [-1, 1]
            )

        def forward(self, x):
            return self.regressor(self.features(x)).squeeze(1)

    return SteeringCNN()


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    session_dirs: list[str],
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.15,
    progress_cb=None,
) -> str:
    """
    Train the steering CNN on *session_dirs* and export to ONNX.

    progress_cb(epoch, total_epochs, train_loss, val_loss) is called after
    each epoch so the Flask server can emit Socket.IO updates.

    Returns the path to the saved ONNX file.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    device = torch.device("cpu")   # Pi 4 has no CUDA

    # ── Data ──────────────────────────────────────────────────────────────────
    X, y = _load_dataset(session_dirs)
    X_t  = torch.from_numpy(X)
    y_t  = torch.from_numpy(y)
    full_ds = TensorDataset(X_t, y_t)

    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = _build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val  = float("inf")
    best_state = None

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        scheduler.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if progress_cb:
            progress_cb(epoch, epochs, train_loss, val_loss)

        print(f"[bc] epoch {epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # ── Export ONNX ──────────────────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_path = os.path.join(MODELS_DIR, f"steering_{ts}.onnx")
    dummy     = torch.zeros(1, 1, IMG_H, IMG_W)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"], output_names=["turn"],
        dynamic_axes={"image": {0: "batch"}},
        opset_version=11,
        dynamo=False,   # forțează exporterul legacy (nu necesită onnxscript)
    )
    print(f"[bc] model saved → {onnx_path}  (best val MSE: {best_val:.4f})")
    return onnx_path


# ══════════════════════════════════════════════════════════════════════════════
# Inference (runs on Pi with OpenCV DNN — no PyTorch needed at runtime)
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralDriver:
    """
    Loads a steering ONNX model and predicts turn values from camera frames.
    Used by the lane-assist thread when behavioral_mode is active.
    """

    def __init__(self, model_path: str):
        self._net = cv2.dnn.readNetFromONNX(model_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[bc] loaded model: {model_path}")

    def predict(self, bgr_frame: np.ndarray) -> float:
        """Returns a turn value ∈ [-1, 1]."""
        img  = _preprocess(bgr_frame)                   # (H, W) float32
        blob = cv2.dnn.blobFromImage(img)               # (1, 1, H, W)
        self._net.setInput(blob)
        out  = self._net.forward()
        return float(np.clip(out.flatten()[0], -1.0, 1.0))
