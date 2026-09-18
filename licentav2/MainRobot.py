from MotorModule import MotorModule
from LaneModule import getLaneCurve
import cv2

# Initialize your custom 4-wheel motors
motor = MotorModule()

def main():
    # Initialize the Camera (0 is usually the default Pi Camera or USB Webcam)
    cap = cv2.VideoCapture(0)
    
    # Set the camera resolution (Murtaza uses 480x240 for faster processing)
    cap.set(3, 480) 
    cap.set(4, 240) 

    # --- TUNING VARIABLES ---
    # Change this to make the car go faster or slower (0.0 to 1.0)
    baseSpeed = 0.45 
    
    # Limits how sharply the car is allowed to turn
    maxTurn = 0.6 
    
    print("🤖 Autonomous Mode Starting...")
    print("Press 'Q' in the video window to stop.")

    while True:
        success, img = cap.read()
        
        if not success:
            print("Failed to read from camera. Check connections!")
            break

        # 1. Get the curve value from the image
        # display=2 shows all the processing windows. Change to 0 when actually driving to save CPU.
        curveVal = getLaneCurve(img, display=2)
        
        # 2. Safety Clamp: Don't let the turn value exceed our maxTurn limit
        if curveVal > maxTurn: curveVal = maxTurn
        if curveVal < -maxTurn: curveVal = -maxTurn

        # 3. Drive the Motors!
        # If the curve is 0, it goes straight. If curve is positive, it turns right.
        motor.move(speed=baseSpeed, turn=curveVal)

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            motor.stop()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()