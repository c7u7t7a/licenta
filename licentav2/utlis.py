import cv2
import numpy as np


# ── CLAHE instance (created once, reused every frame) ─────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def thresholding(img):
    """
    Original thresholding — kept for compatibility.
    Use thresholding_robust() for better results with a low-quality camera.
    """
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([80, 0, 0])
    upperWhite = np.array([255, 160, 255])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    return maskWhite


def thresholding_course(img: np.ndarray) -> np.ndarray:
    """Thresholding original din cursul computervision.zone — simplu și eficient."""
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([80,  0,   0])
    upperWhite = np.array([255, 160, 255])
    return cv2.inRange(imgHsv, lowerWhite, upperWhite)


def thresholding_robust(img: np.ndarray) -> np.ndarray:
    """
    Detectare drum (foi negre/gri) pe imaginea DEJA WARPATĂ.

    Drumul = foi negre/gri, delimitat de bandă galbenă.
    Suprafețe albe/luminoase (foi albe, reflexii, podea deschisă) sunt EXCLUSE.

    Pași:
      1. Blur          — reduce zgomotul
      2. mask_valid    — V > 20  (exclude bordul negru pur al warp-ului)
      3. mask_dark     — V < 110 (excludem suprafețele luminoase: foi albe, podea deschisă)
      4. mask_yellow   — exclude liniile galbene de margine
      5. mask = valid AND dark AND NOT yellow
      6. CLOSE         — umple goluri mici
      7. Largest blob  — suprafața de drum principală
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Pixeli valizi din warp: V > 20 (exclude bordul negru pur al warp-ului)
    mask_valid  = cv2.threshold(v, 20, 255, cv2.THRESH_BINARY)[1]

    # Pixeli întunecați = suprafață drum: V < 150
    # Exclude foi albe (V≈200+), podea deschisă, reflexii
    mask_dark   = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # Exclude liniile galbene de margine
    mask_yellow = cv2.inRange(hsv,
                               np.array([10, 100,  80]),
                               np.array([45, 255, 255]))

    # Drum = pixeli valizi, întunecați și care nu sunt galbeni
    mask = cv2.bitwise_and(mask_valid, mask_dark)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_yellow))

    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Largest blob = suprafața drumului
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas)) + 1
    if areas[best - 1] < 500:
        return np.zeros_like(mask)

    return np.where(labels == best, np.uint8(255), np.uint8(0))


def thresholding_road(img: np.ndarray) -> np.ndarray:
    """
    Detectează foaia neagră (drumul) dintre liniile galbene folosind flood fill.

    Algoritmul:
      1. Detectează liniile galbene → bariere care delimitează drumul
      2. Dilată barierele pentru a închide goluri mici
      3. Mască candidat = pixeli valizi (V > 15) și non-galbeni
      4. Flood fill din centrul de jos al imaginii → se oprește la barierele galbene
         → izolează exact zona de drum dintre linii
      5. Morphology close → umple goluri mici din foaia neagră

    Rezultat: 255 = drum (foaie neagră), 0 = tot restul (podea, pereți, afara liniilor)
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h_img, w_img = img.shape[:2]

    # ── 1. Bariere galbene (liniile de margine) ───────────────────────────────
    mask_yellow = cv2.inRange(hsv,
                               np.array([10, 100,  80]),
                               np.array([45, 255, 255]))
    ky = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_yellow = cv2.dilate(mask_yellow, ky, iterations=2)

    # ── 2. Zonă validă (exclude bordul negru pur al warp-ului, V > 15) ────────
    mask_valid = cv2.threshold(hsv[:, :, 2], 15, 255, cv2.THRESH_BINARY)[1]

    # ── 3. Candidat = valid și NU galben ──────────────────────────────────────
    mask_cand = cv2.bitwise_and(mask_valid, cv2.bitwise_not(mask_yellow))

    # ── 4. Flood fill din centrul de jos ─────────────────────────────────────
    flood     = mask_cand.copy()
    fill_mask = np.zeros((h_img + 2, w_img + 2), np.uint8)

    seeded = False
    # Caută un pixel valid pornind din centrul de jos și extinzând radial
    for dy in range(0, h_img // 2):
        sy = h_img - 1 - dy
        for dx in range(0, w_img // 2, 4):
            for sx in [w_img // 2 + dx, w_img // 2 - dx]:
                if 0 <= sx < w_img and flood[sy, sx] == 255:
                    cv2.floodFill(flood, fill_mask, (sx, sy), 128)
                    seeded = True
                    break
            if seeded:
                break
        if seeded:
            break

    if not seeded:
        return np.zeros((h_img, w_img), dtype=np.uint8)

    result = np.where(flood == 128, np.uint8(255), np.uint8(0))

    # ── 5. Morphology close pentru a umple goluri mici ────────────────────────
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k, iterations=2)

    return result


def get_lane_center_from_lines(warped: np.ndarray) -> tuple[int | None, int | None, int | None]:
    """
    Găsește centrul benzii din imaginea bird's-eye warpată cu linii galbene.

    Împarte imaginea în jumătatea stângă și dreaptă, găsește centroidul
    fiecărei linii separat, returnează centrul dintre ele.

    Returns: (center_x, left_x, right_x)  — None dacă linia nu e vizibilă
    """
    h, w = warped.shape[:2]
    half = w // 2

    left_px  = warped[:, :half]
    right_px = warped[:, half:]

    # Centroid orizontal al fiecărei jumătăți
    left_cols  = np.where(left_px.sum(axis=0) > 0)[0]
    right_cols = np.where(right_px.sum(axis=0) > 0)[0]

    left_x  = int(np.mean(left_cols))           if left_cols.size  > 0 else None
    right_x = int(np.mean(right_cols)) + half   if right_cols.size > 0 else None

    if left_x is not None and right_x is not None:
        center = (left_x + right_x) // 2
    elif left_x is not None:
        center = left_x + half          # estimare dacă lipsește dreapta
    elif right_x is not None:
        center = right_x - half         # estimare dacă lipsește stânga
    else:
        center = None

    return center, left_x, right_x


def warpImg(img, points, w, h, inv=False):
    pts1   = np.float32(points)
    pts2   = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts2, pts1) if inv else cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h))


def nothing(a):
    pass


def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top",    "Trackbars", intialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top",   "Trackbars", intialTracbarVals[1], hT,      nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom","Trackbars", intialTracbarVals[3], hT,      nothing)


def valTrackbars(wT=480, hT=240):
    widthTop    = cv2.getTrackbarPos("Width Top",     "Trackbars")
    heightTop   = cv2.getTrackbarPos("Height Top",    "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom",  "Trackbars")
    heightBottom= cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([
        (widthTop,       heightTop),
        (wT - widthTop,  heightTop),
        (widthBottom,    heightBottom),
        (wT - widthBottom, heightBottom),
    ])
    return points


def drawPoints(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


def getHistogram(img, minPer=0.1, display=False, region=1):
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0] // region:, :], axis=0)

    maxValue = np.max(histValues)
    minValue = minPer * maxValue

    indexArray = np.where(histValues >= minValue)
    basePoint  = int(np.average(indexArray)) if indexArray[0].size > 0 else img.shape[1] // 2

    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            cv2.line(imgHist, (x, img.shape[0]),
                     (x, img.shape[0] - intensity // 255 // region), (255, 0, 255), 1)
        cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist

    return basePoint


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width  = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],
                                                 (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                 None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x],
                                         (imgArray[0].shape[1], imgArray[0].shape[0]),
                                         None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)
    return ver
