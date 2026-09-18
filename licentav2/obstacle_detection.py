"""
Obstacle Detection — OpenCV-only, "object on the road" approach.
==============================================================

Vechea metodă „tot ce nu e fundal = obstacol" marca drumul întreg ca obstacol.
Metoda nouă caută OBIECTE pe drum (ex. o piatră), nu drumul:

  1. ROI = banda centrală din față (unde merge mașina).
  2. Găsește DRUMUL = cea mai mare regiune ÎNTUNECATĂ (asfaltul negru).
  3. Caută DOAR în interiorul drumului blob-uri care NU sunt asfalt și NU sunt
     bandă galbenă → adică obiecte aflate PE drum (piatra, un obstacol).
  4. Filtre: dimensiune (nici prea mic = zgomot, nici uriaș = drum), compactitate
     (solidity) și raport laturi → un obiect real e un blob compact, nu o fâșie.

De ce merge: drumul (negru) și banda (galben) devin fundal prin construcție.
Un obiect deschis la culoare pe asfaltul negru iese clar în evidență, iar
fundalul alb din lateral e ignorat (e în afara regiunii de drum).

Returnează:
    {detected, bbox(x,y,w,h) în cadru complet, area_frac, distance_frac, dangerous}
"""

import cv2
import numpy as np

# ── ROI: cât din cadru analizăm (drumul din față, inclusiv ce e mai departe) ───
ROI_TOP_FRAC   = 0.15   # include și drumul depărtat (obiecte mai din față)
ROI_BOT_FRAC   = 0.98
ROI_LEFT_FRAC  = 0.05
ROI_RIGHT_FRAC = 0.95

# ── Segmentarea drumului / obiectelor ──────────────────────────────────────────
ROAD_V_MAX     = 95     # asfaltul e mai întunecat decât atât (canalul V din HSV)
OBJ_V_MARGIN   = 25     # un obiect e mai luminos decât ROAD_V_MAX + atât
MIN_ROAD_FRAC  = 0.06   # avem nevoie de un drum vizibil (≥6% din ROI) ca să căutăm

# ── Filtre pe obiect ────────────────────────────────────────────────────────────
MIN_AREA_PX    = 150    # cel mai mic obiect acceptat (px²) — piatra e mică/depărtată
MIN_SOLIDITY   = 0.45   # area/convex_hull → blob compact, nu fâșii împrăștiate
MIN_ASPECT     = 0.30
MAX_ASPECT     = 3.5

# ── Praguri de raportare ────────────────────────────────────────────────────────
# Obiectele sunt mici → filtrul principal de zgomot e MIN_AREA_PX (absolut),
# nu fracția. DETECT mic ca să prindem și o piatră depărtată.
DETECT_THRESHOLD = 0.0010  # area obiect / area ROI → „prezent"
DANGER_THRESHOLD = 0.010   # area obiect / area ROI → „aproape / periculos" → stop
CONFIRM_FRAMES   = 2       # cadre consecutive necesare

# ── Culori de exclus din obiecte (bandă + SEMNE de circulație) ─────────────────
# Un semn rutier (roșu/albastru saturat) NU e obstacol pe drum.
_YELLOW_LO = np.array([ 15,  40,  40], dtype=np.uint8)
_YELLOW_HI = np.array([ 75, 255, 255], dtype=np.uint8)
_RED1_LO   = np.array([  0,  80,  60], dtype=np.uint8)
_RED1_HI   = np.array([ 10, 255, 255], dtype=np.uint8)
_RED2_LO   = np.array([160,  80,  60], dtype=np.uint8)
_RED2_HI   = np.array([179, 255, 255], dtype=np.uint8)
_BLUE_LO   = np.array([ 95,  80,  50], dtype=np.uint8)
_BLUE_HI   = np.array([135, 255, 255], dtype=np.uint8)

# Cât „umflăm" caseta unui semn detectat înainte să excludem zona (pol/suport).
# Mic, ca să nu înghită un obstacol real aflat lângă semn.
SIGN_BOX_PAD = 0.15

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


class ObstacleDetector:
    def __init__(self):
        self._confirm = 0

    def reset(self):
        """Clear the confirmation counter (call when detection is toggled off)."""
        self._confirm = 0

    def detect(self, frame: np.ndarray, exclude_boxes=None) -> dict:
        """
        Analyse *frame* (BGR) and return obstacle information.

        exclude_boxes: listă opțională de casete de SEMNE (x1, y1, x2, y2) în
        coordonate de cadru complet. Orice blob care se suprapune cu o casetă de
        semn (umflată cu SIGN_BOX_PAD) e ignorat → semnele nu devin obstacole.
        """
        h, w = frame.shape[:2]
        y0, y1 = int(h * ROI_TOP_FRAC), int(h * ROI_BOT_FRAC)
        x0, x1 = int(w * ROI_LEFT_FRAC), int(w * ROI_RIGHT_FRAC)
        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        roi_area = rh * rw
        if roi_area == 0:
            return _empty()

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v   = hsv[:, :, 2]

        # ── 1. Drumul = cea mai mare regiune întunecată ──────────────────────────
        dark = cv2.inRange(v, 0, ROAD_V_MAX)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, _K7, iterations=2)
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            self._confirm = 0
            return _empty()
        road_c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(road_c) < MIN_ROAD_FRAC * roi_area:
            # niciun drum clar în față → nu putem izola obiecte
            self._confirm = 0
            return _empty()
        road_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(road_mask, [road_c], -1, 255, cv2.FILLED)

        # ── 2. Obiecte = pete luminoase PE drum, fără culori de bandă/SEMN ───────
        yellow = cv2.inRange(hsv, _YELLOW_LO, _YELLOW_HI)
        red    = cv2.bitwise_or(cv2.inRange(hsv, _RED1_LO, _RED1_HI),
                                cv2.inRange(hsv, _RED2_LO, _RED2_HI))
        blue   = cv2.inRange(hsv, _BLUE_LO, _BLUE_HI)
        sign_col = cv2.bitwise_or(cv2.bitwise_or(yellow, red), blue)
        bright = cv2.inRange(v, ROAD_V_MAX + OBJ_V_MARGIN, 255)
        obj = cv2.bitwise_and(road_mask, bright)
        obj = cv2.bitwise_and(obj, cv2.bitwise_not(sign_col))
        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN,  _K3, iterations=1)
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, _K5, iterations=1)

        # Casete de semne (în coord. cadru complet) → umflate, pt. excludere
        excl = _inflate_boxes(exclude_boxes, SIGN_BOX_PAD) if exclude_boxes else []

        # ── 3. Cel mai mare blob COMPACT valid ───────────────────────────────────
        ocnts, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = 0
        best_bbox = None
        for c in ocnts:
            area = cv2.contourArea(c)
            if area < MIN_AREA_PX:
                continue
            x, y, wc, hc = cv2.boundingRect(c)
            aspect = wc / max(hc, 1)
            if not (MIN_ASPECT < aspect < MAX_ASPECT):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity  = area / hull_area if hull_area > 0 else 0.0
            if solidity < MIN_SOLIDITY:
                continue
            # Ignoră blob-ul dacă se suprapune cu o casetă de SEMN detectat
            full_box = (x0 + x, y0 + y, x0 + x + wc, y0 + y + hc)
            if any(_overlaps(full_box, e) for e in excl):
                continue
            if area > best_area:
                best_area = area
                best_bbox = (x, y, wc, hc)

        if best_bbox is None:
            self._confirm = 0
            return _empty()

        area_frac = best_area / roi_area
        if area_frac < DETECT_THRESHOLD:
            self._confirm = 0
            return _empty(area_frac=round(area_frac, 4))

        # ── Confirmare ───────────────────────────────────────────────────────────
        self._confirm = min(self._confirm + 1, CONFIRM_FRAMES * 4)
        if self._confirm < CONFIRM_FRAMES:
            return _empty(area_frac=round(area_frac, 4))

        rx, ry, rw_, rh_ = best_bbox
        full_bbox = (x0 + rx, y0 + ry, rw_, rh_)
        distance_frac = min(1.0, area_frac / DANGER_THRESHOLD)
        return {
            "detected":      True,
            "bbox":          full_bbox,
            "area_frac":     round(area_frac, 4),
            "distance_frac": round(distance_frac, 3),
            "dangerous":     area_frac >= DANGER_THRESHOLD,
        }

    @staticmethod
    def draw(frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw obstacle bounding box + proximity bar onto *frame* (copy)."""
        if not result.get("detected"):
            return frame
        out = frame.copy()
        x, y, w, h = result["bbox"]
        danger = result.get("dangerous", False)
        color  = (0, 0, 230) if danger else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"OBSTACOL {'!' if danger else ''} {result['area_frac']*100:.1f}%"
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        bar_w = int(100 * result["distance_frac"])
        cv2.rectangle(out, (x, y + h + 4), (x + bar_w, y + h + 12), color, -1)
        return out


def _empty(area_frac: float = 0.0) -> dict:
    return {"detected": False, "bbox": None, "area_frac": area_frac, "distance_frac": 0.0}


def _inflate_boxes(boxes, pad_frac: float):
    """Umflă fiecare casetă (x1,y1,x2,y2) cu pad_frac din dimensiunea ei."""
    out = []
    for b in boxes:
        try:
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        except (TypeError, IndexError):
            continue
        pw = (x2 - x1) * pad_frac
        ph = (y2 - y1) * pad_frac
        out.append((x1 - pw, y1 - ph, x2 + pw, y2 + ph))
    return out


def _overlaps(a, b) -> bool:
    """True dacă două casete (x1,y1,x2,y2) se suprapun (orice arie comună)."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return ix > 0 and iy > 0
