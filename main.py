# print("I am Death the Destroyer of worlds!!!")

import cv2
import time
import numpy as np
import math
import ctypes
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandTrackingModule as htm
import winsound

# Camera dimensions
wCam, hCam = 1280, 720

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.handDetector(min_detection_confidence=0.7)

# Access the system audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get the volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Fixed volume increments
volumeIncrement = 2.0
heldVolumeIncrement = 0.2  # Reduced increment for held volume changes
volumeDecrement = -2.0
heldVolumeDecrement = -0.2  # Reduced decrement for held volume changes

# Feature to enable or disable beep sound
beepSoundFeature = True  # Set to False to disable beep sound

pTime = 0  # Previous time for FPS calculation
fingersTouching = False  # Flag to track if fingers are touching
touchStartTime = 0  # Time when fingers first touch
beepPlayed = False  # Flag to ensure beep plays only once

# Distance threshold to recognize finger taps/holds
distanceThreshold = 150  # Adjust this value based on your camera and preference


# Function to simulate key press for volume control
def simulate_volume_key_press(key_code):
    # Define necessary structures for key press simulation
    PUL = ctypes.POINTER(ctypes.c_ulong)

    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort),
                    ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]

    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong),
                    ("wParamL", ctypes.c_short),
                    ("wParamH", ctypes.c_ushort)]

    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]

    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput),
                    ("mi", MouseInput),
                    ("hi", HardwareInput)]

    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong),
                    ("ii", Input_I)]

    # Define key press event
    def press_key(hex_key_code):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(wVk=hex_key_code, wScan=0, dwFlags=0, time=0, dwExtraInfo=ctypes.pointer(extra))
        x = Input(type=1, ii=ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    # Define key release event
    def release_key(hex_key_code):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(wVk=hex_key_code, wScan=0, dwFlags=2, time=0, dwExtraInfo=ctypes.pointer(extra))
        x = Input(type=1, ii=ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    # Press and release the key
    press_key(key_code)
    time.sleep(0.05)
    release_key(key_code)


while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Find hands in the image
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        hand_label = detector.handType()

        if hand_label == "Right" or hand_label == "Left":
            # Get coordinates for thumb tip (ID 4) and index finger tip (ID 8)
            x1, y1, z1 = lmList[4][1], lmList[4][2], lmList[4][3]
            x2, y2, z2 = lmList[8][1], lmList[8][2], lmList[8][3]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw circles and line
            cv2.circle(img, (x1, y1), 15, (255, 0, 23), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 23), cv2.FILLED)
            cv2.circle(img, (cx, cy), 15, (255, 0, 23), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 23), 3)

            # Calculate the 2D distance between the thumb and index finger
            length2D = math.hypot(x2 - x1, y2 - y1)

            # Check if the hand is close enough to the camera
            if z1 < distanceThreshold and z2 < distanceThreshold:
                # Check if thumb and index finger tips are touching
                if length2D < 30:  # Adjust threshold as needed
                    if not fingersTouching:
                        touchStartTime = time.time()
                        fingersTouching = True
                        beepPlayed = False
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

                        if hand_label == "Right":
                            # Increment the volume by 2
                            currentVolume = volume.GetMasterVolumeLevelScalar()
                            newVolume = min(currentVolume + volumeIncrement / 100.0, 1.0)
                            volume.SetMasterVolumeLevelScalar(newVolume, None)
                            # Simulate volume up key press
                            simulate_volume_key_press(0xAF)
                        elif hand_label == "Left":
                            # Decrement the volume by 2
                            currentVolume = volume.GetMasterVolumeLevelScalar()
                            newVolume = max(currentVolume + volumeDecrement / 100.0, 0.0)
                            volume.SetMasterVolumeLevelScalar(newVolume, None)
                            # Simulate volume down key press
                            simulate_volume_key_press(0xAE)
                    else:
                        if time.time() - touchStartTime > 1.5:  # Hold duration threshold (1.5 seconds)
                            if not beepPlayed and beepSoundFeature:
                                winsound.Beep(1000, 200)  # Beep sound when held together
                                beepPlayed = True
                            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                            if hand_label == "Right":
                                # Increment the volume by a smaller amount
                                currentVolume = volume.GetMasterVolumeLevelScalar()
                                newVolume = min(currentVolume + heldVolumeIncrement / 100.0, 1.0)
                                volume.SetMasterVolumeLevelScalar(newVolume, None)
                                # Simulate volume up key press
                                simulate_volume_key_press(0xAF)
                            elif hand_label == "Left":
                                # Decrement the volume by a smaller amount
                                currentVolume = volume.GetMasterVolumeLevelScalar()
                                newVolume = max(currentVolume + heldVolumeDecrement / 100.0, 0.0)
                                volume.SetMasterVolumeLevelScalar(newVolume, None)
                                # Simulate volume down key press
                                simulate_volume_key_press(0xAE)
                else:
                    fingersTouching = False  # Reset the flag when fingers are not touching
                    beepPlayed = False
                    cv2.circle(img, (cx, cy), 15, (255, 0, 23), cv2.FILLED)

    # Get the current volume level for the volume bar
    currentVolume = volume.GetMasterVolumeLevelScalar()
    volBar = np.interp(currentVolume, [0, 1], [400, 150])  # Volume bar height mapping
    volPer = np.interp(currentVolume, [0, 1], [0, 100])  # Volume percentage

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)

    # Display the image in a window
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

