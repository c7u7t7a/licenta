import cv2
import numpy as np
import utlis # Make sure your utlis.py file is in the same folder!

curveList = []
avgVal = 10

def getLaneCurve(img, display=2):
    imgCopy = img.copy()
    imgResult = img.copy()
    
    # STEP 1: Get Threshold Image (Finds the white/paper paths)
    imgThres = utlis.thresholding(img)
    
    # STEP 2: Warp Image (Bird's Eye View)
    hT, wT, c = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)
    
    # STEP 3: Find Histogram & Curve
    middlePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint
    
    # STEP 4: Average out the curve (Smooths out the steering so it doesn't jitter)
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    
    # STEP 5: Display the visual pipeline windows
    if display != 0:
        imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT//3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        
        midY = 450
        cv2.putText(imgResult, str(curve), (wT//2-80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT//2, midY), (wT//2+(curve*3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve//50), midY-10),
                     (w * x + int(curve//50), midY+10), (0, 0, 255), 2)
            
    if display == 2:
        # Shows all the processing steps in one big window
        imgStacked = utlis.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        # Shows just the final result
        cv2.imshow('Result', imgResult)
        
    # STEP 6: Normalization (Convert curve value to a number between -1 and 1 for the motors)
    curve = curve/100
    if curve > 1.0: curve = 1.0
    if curve < -1.0: curve = -1.0
        
    return curve

if __name__ == '__main__':
    # This block is just for testing the camera without running the motors
    cap = cv2.VideoCapture(0) # '0' uses the connected Raspberry Pi camera or USB webcam
    
    # You will need to adjust these values in the course to perfectly match your track!
    initialTrackBarVals = [102, 80, 20, 214] 
    utlis.initializeTrackbars(initialTrackBarVals)
    
    while True:
        success, img = cap.read()
        if success:
            img = cv2.resize(img, (480, 240))
            curve = getLaneCurve(img, display=2)
            print("Curve Value for Motors:", curve)
            cv2.waitKey(1)
        else:
            print("Camera not found!")
            break