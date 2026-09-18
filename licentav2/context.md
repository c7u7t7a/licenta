# Mașină Autonomă pe Raspberry Pi 4 — Context complet pentru prezentare PowerPoint

> Acest fișier conține TOT proiectul, structurat ca să poată fi transformat
> direct într-o prezentare PowerPoint. La final există și un **outline de
> slide-uri** sugerat. Limba: română (proiect de licență).

---

## 1. Rezumat (elevator pitch)

Mașină autonomă la scară mică, construită pe **Raspberry Pi 4**, care:
- **urmărește banda** (lane keeping) folosind camera și OpenCV + un regulator PID;
- **recunoaște semne de circulație** (STOP, CEDEAZĂ, limite de viteză 30/50/90/100)
  cu o rețea neuronală ONNX (GTSRB) și le **respectă** (oprire, încetinire, limită viteză);
- **detectează obstacole** pe drum cu camera (ex. o piatră) și se oprește;
- **frânează de urgență** la coliziune cu senzori ultrasonici;
- **parchează automat** cu fața în garaj (rotire 180° + intrare dreaptă);
- e controlată complet dintr-o **interfață web modernă** (dashboard live, dark/light mode).

Totul rulează local pe Pi, fără cloud, fără GPU.

---

## 2. Obiective

- Demonstrarea unui **pipeline complet de conducere autonomă** pe hardware ieftin.
- Integrarea **viziunii computerizate clasice** (OpenCV) cu **deep learning** (ONNX).
- **Fuziune de senzori**: cameră + 6 senzori ultrasonici.
- Interfață de **telemetrie și control în timp real** (WebSocket).
- Robustețe la condiții reale: lumină slabă, senzori defecți, zgomot.

---

## 3. Stack tehnologic

| Componentă | Tehnologie |
|---|---|
| Limbaj | Python 3.11 |
| Server web | Flask + Flask-SocketIO (WebSocket) |
| Viziune | OpenCV (cv2) |
| Deep learning | ONNX Runtime via `cv2.dnn` (MobileNetV2, GTSRB) |
| Control motoare | gpiozero (RPi.GPIO / rpi-lgpio) |
| Senzori | gpiozero `DistanceSensor` (HC-SR04) |
| Cameră | Picamera2 / libcamera (IMX708 – Camera Module 3) |
| Frontend | HTML + CSS + JS, Chart.js, nipplejs (joystick), Socket.IO |
| Gamepad | pydualsense / evdev (DS4 opțional) |
| Pornire automată | systemd service |

---

## 4. Hardware

### Raspberry Pi 4 (Raspberry Pi OS Bookworm)
- Server pornit automat la boot prin **systemd** (`robot.service`).

### Tracțiune — 4 motoare DC (skid-steer / tank drive)
| Motor | Forward | Backward |
|---|---|---|
| front_left | GPIO 17 | GPIO 27 |
| front_right | GPIO 22 | GPIO 23 |
| rear_left | GPIO 5 | GPIO 6 |
| rear_right | GPIO 13 | GPIO 19 |

- Control diferențial: `move(speed, turn)` → `left = speed+turn`, `right = speed-turn`.
- Skid-steer → se poate roti **pe loc** (esențial pentru parcare).

### Cameră — Raspberry Pi Camera Module 3 (IMX708)
- 640×480, flux MJPEG către browser.
- Montată sus, înclinată ~20–30° în jos (vede ambele benzi galbene).

### 6× Senzori ultrasonici HC-SR04
- 3 față (front_left/center/right) + 3 spate (rear_left/center/right).
- Folosiți pentru **frânare de urgență** (cei 3 din față) și **parcare** (toți).
- ⚠️ 2 senzori arși/defecți pe placa fizică: `rear_center` + `front_center`
  (citesc valori random) → ignorați în software.

---

## 5. Arhitectura software

### Fișiere principale
```
app.py                 # Server Flask + SocketIO; orchestrează toate firele de execuție
utlis.py               # Funcții OpenCV (thresholding, warp, histogramă)
pid.py                 # Regulator PID thread-safe (lane keeping)
MotorModule.py         # Control motoare (gpiozero)
UltrasonicModule.py    # Array 6× HC-SR04
sign_detection.py      # Detecție semne (ONNX + shape + red-fill) + DriveBehavior
obstacle_detection.py  # Detecție obstacole cameră ("object on road")
parking_module.py      # Parcare automată (mașină de stări)
behavioral_cloning.py  # Antrenare + inferență CNN (end-to-end driving)
DS4Module.py           # Controller PS4 (opțional)
car_config.json        # Configurație persistentă
traffic_sign_classifier.onnx  # Model semne (GTSRB, 43 clase)
templates/             # 7 pagini web (index, login, analytics, config, race, sessions, training)
```

### Fire de execuție (threads) în `app.py`
Server-ul rulează mai multe bucle paralele, thread-safe (locks):
- **Camera** — citește cadre de la cameră (Picamera2).
- **Lane assist** (20 Hz) — detecție bandă → PID → comenzi motoare.
- **Sign detection** (8 Hz) — rulează modelul ONNX, actualizează comportamentul.
- **Obstacle detection** (10 Hz) — caută obstacole pe drum.
- **Ultrasonic** (15 Hz) — citește senzorii, frânare de urgență.
- **DS4 drive** (30 Hz) — citește gamepad-ul.
- **MJPEG stream** — trimite video în browser.
- **Stats sistem** — temperatură / CPU / RAM.

---

## 6. Funcționalitatea 1 — Lane Keeping (urmărire bandă)

### Pipeline (mod „warp", `_detect_lane` în app.py)
1. **ROI** = banda de jos (de la 40% în jos) — ce e imediat în față.
2. Resize la 320×120 px.
3. **Mască galbenă HSV** `[10,100,80] → [45,255,255]` (S ridicat exclude albul).
4. **Centroid ponderat** per jumătate (stânga/dreapta) după densitatea de galben
   → robust la reflexii izolate; gate minim de pixeli per parte.
5. **Centrul benzii**:
   - ambele linii → media; o singură linie → offset cu boost ×2; niciuna → drept.
6. **Eroare** = centru_bandă − centru_imagine, normalizată agresiv (`/35`).
7. **Netezire temporală** (anti-spazz): median-3 → EMA (α=0.65) → limită de pantă
   (max 0.18/cadru). Elimină salturile când o linie pâlpâie.
8. **PID** transformă eroarea în comandă de viraj.

### Regulator PID (`pid.py`)
- `Kp=0.6, Ki=0.01, Kd=0.25`.
- Anti-windup pe integrator, filtru EMA pe derivată, thread-safe.

### Adaptive Speed (frânare în curbe)
- `viteză = viteză_bază × max(0, 1 − k·|curbă|)`  (k = 0.85).
- În curbă strânsă viteza scade aproape la 0 → nu iese de pe pistă.

---

## 7. Funcționalitatea 2 — Detecție semne rutiere

### Pipeline în 2 etape (`sign_detection.py`)
1. **Etapa 1 — segmentare culoare (OpenCV, ~2 ms):** găsește regiuni roșii/albastre
   (bordura semnelor) în **top 60%** din cadru (jos e pista). Morfologie pentru a
   uni inelul roșu al limitelor de viteză.
2. **Etapa 2 — clasificator ONNX (MobileNetV2, GTSRB, 43 clase):** clasifică fiecare
   regiune candidat. Rulează pe CPU.

### Corecții inteligente (fără reantrenare)
- **Shape-based**: forma conturului (cerc / triunghi / octogon) corectează confuzia
  cerc↔triunghi (limită de viteză citită ca „cedează").
- **Red-fill pentru STOP**: un octogon are circularitate ~0.95 → citit ca „cerc".
  Diferența: STOP e **umplut cu roșu**, limita de viteză e un **inel roșu cu centru
  alb**. Se măsoară fracția de roșu → STOP detectat corect.
- **Anti-fals YIELD**: o clasă triunghi e acceptată DOAR dacă și conturul e triunghi.

### Respectarea semnelor (`DriveBehavior` + app.py)
- **STOP** → oprire ~2.5 s, apoi reia.
- **CEDEAZĂ** → încetinire temporară.
- **Limite de viteză** (30/50/90/100) → mașina **merge LA viteza din semn** (valori
  absolute diferențiate: 0.40 / 0.47 / 0.53 / 0.58), nu doar plafon. Semnul rămâne
  **memorat** până la alt semn de viteză.
- Notă: GTSRB n-are clasa „90" → `speed_limit_80` remapat ca 90 km/h.

---

## 8. Funcționalitatea 3 — Detecție obstacole (cameră)

### Abordare „obiect pe drum" (`obstacle_detection.py`)
Pista: drum NEGRU în centru, bandă galbenă pe laturi, podea albă.
1. **Găsește drumul** = cea mai mare regiune întunecată (asfaltul).
2. **Caută obiecte DOAR pe drum** = pete luminoase care nu sunt bandă galbenă
   (ex. o piatră deschisă la culoare pe asfaltul negru).
3. **Filtre**: dimensiune minimă, compactitate (solidity), raport laturi.
4. **Exclude semnele**: culorile de semn (roșu/albastru saturat) + casetele
   semnelor detectate → un semn de circulație nu e luat ca obstacol.
- Oprirea sigură reală pe mers înainte vine de la **senzorii ultrasonici față**;
  camera = ajutor vizual.

---

## 9. Funcționalitatea 4 — Senzori ultrasonici (fuziune)

- **Frânare de urgență**: cei 3 senzori din **față**; dacă cel mai apropiat < 30 cm
  → oprire imediată motoare (blochează lane assist până se eliberează calea).
- Citiri filtrate, `partial=True` (gpiozero) ca să nu blocheze firul dacă un senzor
  nu primește ecou.
- Grilă live a celor 6 senzori în interfața web (color-coded după distanță).

---

## 10. Funcționalitatea 5 — Parcare automată (nose-in)

### Scenariu
Mașina pornește **întoarsă invers** (cu spatele spre garaj de carton cu 3 pereți).
Manevra are DOAR două mișcări — **simplu și fiabil**:
1. **TURN** — se rotește **180° pe loc** (bazat pe timp: `TURN_TIME_S=1.7s`,
   `TURN_RATE=0.8`). Acum fața e spre garaj.
2. **FORWARD** — merge **DREPT înainte** (fără viraje) până când **oricare** senzor
   față valid e ≤ **25 cm** de perete → STOP.

### Robustețe
- Senzorii defecți (`rear_center`, `front_center` – arși) sunt **ignorați complet**.
- **Filtru median** per senzor → respinge valorile random.
- Oprirea folosește `min(front_left, front_right)` → robustă.

---

## 11. Alte funcționalități

- **Mod manual**: joystick virtual (touch), tastatură (WASD + Space=STOP), gamepad PS4.
- **Record & Playback**: înregistrează comenzile manuale (20 Hz) și le redă în buclă.
- **Behavioral Cloning**: rețea CNN antrenată să prezică virajul direct din imagine
  (end-to-end), ca alternativă la pipeline-ul clasic.
- **Race Timer**: cronometru ture cu „best lap".
- **Analytics**: grafice și statistici din sesiunile înregistrate.
- **Config Manager**: editare parametri (viteză, PID, warp, ROI) din UI, cu preseturi.

---

## 12. Interfața web (dashboard)

- **7 pagini**: Dashboard (index), Login, Analytics, Config, Race, Sessions, Training.
- **Live**: flux video MJPEG cu suprapuneri (bandă, semne, obstacole, săgeată viraj).
- **Telemetrie în timp real** (WebSocket): viteză, viraj, curbă, contribuții PID,
  încredere bandă, distanțe senzori, traiectorie estimată (dead reckoning).
- **Control complet**: Manual / Lane Assist / STOP, toggle Semne / Obstacole /
  Crash Det. / Parcare Auto, slidere viteză și parametri.
- **Dark / Light mode** pe toate paginile (temă partajată, salvată în browser).
- Responsive (desktop + mobil).

---

## 13. Probleme rezolvate (decizii de inginerie — bune pentru slide „Provocări")

| Provocare | Soluție |
|---|---|
| Camera IMX708 inversa culorile | Picamera2 + conversie YUV420→RGB corectă |
| Detecția prindea albul/cerul ca semn | Eliminat masca albă/galbenă; căutare doar în top 60% |
| STOP nedetectat (octogon citit ca cerc) | Discriminare prin fracția de roșu |
| YIELD detectat random | Acceptă triunghi doar dacă și conturul e triunghi |
| Limita de viteză „memora" dar nu schimba viteza | Viteze absolute țintă, nu plafon `min()` |
| Lane assist „spazzing" (pâlpâit linii) | Filtru median-3 + EMA + limită de pantă |
| Detecția de obstacole marca tot drumul | Abordare „obiect pe drum" + excludere semne |
| Senzori ultrasonici blocau firul (fără ecou) | `partial=True` + filtru median, senzori defecți ignorați |
| Parcarea se oprea devreme (senzor ars) | Ignorat senzorii defecți + rotire 180° pe timp |
| Lumină slabă pe cameră | CLAHE pe canalul L (LAB) |

---

## 14. Rezultate / status

- Lane keeping + PID + adaptive speed: **funcțional pe pistă**.
- Detecție semne: **funcțională** (STOP, CEDEAZĂ, 30/50/90/100), respectate.
- Detecție obstacole cameră + ultrasonic: **funcțional**.
- Parcare automată nose-in: **implementată**, în calibrare pe mașina reală
  (`TURN_TIME_S`, sensul rotirii).
- Interfață web completă cu dark/light mode pe toate paginile.

---

## 15. OUTLINE SUGERAT DE SLIDE-URI (pentru PowerPoint)

1. **Titlu** — „Mașină autonomă pe Raspberry Pi 4" + nume, coordonator, an.
2. **Motivație & obiective** — de ce, ce demonstrăm.
3. **Privire de ansamblu** — schema mașinii + ce știe să facă (5 bullet-uri).
4. **Hardware** — Pi 4, motoare, cameră, 6× ultrasonic (poză/diagramă).
5. **Arhitectură software** — fișiere + fire de execuție paralele (diagramă).
6. **Stack tehnologic** — tabelul de tehnologii.
7. **Lane keeping** — pipeline-ul pe pași (cu imagine din cameră).
8. **PID & Adaptive speed** — formule + grafic viteză vs. curbă.
9. **Detecție semne** — cele 2 etape + corecțiile inteligente (cu poze semne).
10. **Respectarea semnelor** — STOP / CEDEAZĂ / limite de viteză (tabel).
11. **Detecție obstacole** — „obiect pe drum" (înainte/după).
12. **Senzori ultrasonici** — fuziune + frânare de urgență.
13. **Parcare automată** — diagrama celor 2 faze (180° + înainte).
14. **Interfața web** — screenshot dashboard (dark + light).
15. **Provocări & soluții** — tabelul de probleme rezolvate (2-3 cele mai tari).
16. **Rezultate / demo** — ce funcționează + (eventual) link video.
17. **Concluzii & dezvoltări viitoare** — ce s-ar putea adăuga.
18. **Q&A / Mulțumiri**.

### Idei vizuale pentru slide-uri
- Diagrama fluxului: Cameră → OpenCV → PID → Motoare.
- Schema fuziunii: Cameră + 6× Ultrasonic → Decizie.
- Before/after pentru detecția de obstacole și pentru light mode.
- Mașină de stări pentru parcare (2 cutii: TURN → FORWARD).
- Grafic „viteză efectivă vs. unghi curbă" pentru adaptive speed.
