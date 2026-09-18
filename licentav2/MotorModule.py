from gpiozero import Motor, Device
from gpiozero.pins.rpigpio import RPiGPIOFactory

# Use RPi.GPIO (provided by rpi-lgpio) so PWM works on any pin
Device.pin_factory = RPiGPIOFactory()

# Motors — exact same pins as the working test script
front_left  = Motor(forward=17, backward=27)
front_right = Motor(forward=22, backward=23)
rear_left   = Motor(forward=5,  backward=6)
rear_right  = Motor(forward=13, backward=19)

# Physical sides (diagonal wiring):
#   LEFT  side = front_left  + rear_right
#   RIGHT side = front_right + rear_left


# ── Movement functions ───────────────────────────────────────────────────────

def drive_forward(speed=1.0):
    front_left.forward(speed)
    rear_left.backward(speed)
    front_right.forward(speed)
    rear_right.backward(speed)

def drive_backward(speed=1.0):
    front_left.backward(speed)
    rear_left.forward(speed)
    front_right.backward(speed)
    rear_right.forward(speed)

def turn_left(speed=1.0):
    # Left side (front_left + rear_right) → backward
    front_left.forward(speed)
    rear_right.backward(speed)
    # Right side (front_right + rear_left) → forward
    front_right.backward(speed)
    rear_left.forward(speed)

def turn_right(speed=1.0):
    # Left side (front_left + rear_right) → forward
    front_left.backward(speed)
    rear_right.forward(speed)
    # Right side (front_right + rear_left) → backward
    front_right.forward(speed)
    rear_left.backward(speed)

def stop_motors():
    front_left.stop()
    rear_left.stop()
    front_right.stop()
    rear_right.stop()


# ── MotorModule wrapper — used by app.py and lane assist ─────────────────────

class MotorModule:
    def move(self, speed=0.0, turn=0.0, t=0):
        """
        speed : -1.0 (reverse) … +1.0 (forward)
        turn  : -1.0 (left)    … +1.0 (right)
        """
        left  = max(min(speed + turn, 1.0), -1.0)
        right = max(min(speed - turn, 1.0), -1.0)

        # LEFT side = front_left + rear_left
        if left > 0:
            front_left.forward(left)
            rear_left.forward(left)
        elif left < 0:
            front_left.backward(abs(left))
            rear_left.backward(abs(left))
        else:
            front_left.stop()
            rear_left.stop()

        # RIGHT side = front_right + rear_right
        if right > 0:
            front_right.forward(right)
            rear_right.forward(right)
        elif right < 0:
            front_right.backward(abs(right))
            rear_right.backward(abs(right))
        else:
            front_right.stop()
            rear_right.stop()

        if t:
            import time; time.sleep(t)
            self.stop()

    def stop(self):
        stop_motors()
