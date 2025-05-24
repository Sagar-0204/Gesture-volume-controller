import cv2
import mediapipe as mp
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]
curr_vol = volume.GetMasterVolumeLevel()

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Camera setup
cap = cv2.VideoCapture(0)
prev_time = 0
vol_smooth = curr_vol
control_active = False
muted = False
last_vol = curr_vol

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = img.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Draw hand
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture to toggle lock (thumb tip & pinky tip)
            if len(lm_list) >= 21:
                thumb_tip = lm_list[4]
                pinky_tip = lm_list[20]
                gesture_distance = np.hypot(pinky_tip[0] - thumb_tip[0], pinky_tip[1] - thumb_tip[1])
                if gesture_distance < 40:
                    control_active = not control_active
                    time.sleep(0.5)

            # Gesture to mute/unmute (index tip & middle tip)
            if len(lm_list) >= 12:
                index_tip = lm_list[8]
                middle_tip = lm_list[12]
                mute_distance = np.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])
                if mute_distance < 30:
                    if not muted:
                        last_vol = vol_smooth
                        volume.SetMasterVolumeLevel(min_vol, None)
                        muted = True
                        time.sleep(0.5)
                    else:
                        volume.SetMasterVolumeLevel(last_vol, None)
                        muted = False
                        time.sleep(0.5)

            # Main volume control (thumb tip & index tip)
            if len(lm_list) >= 9:
                x1, y1 = lm_list[4]
                x2, y2 = lm_list[8]
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                length = np.hypot(x2 - x1, y2 - y1)
                cv2.putText(img, f'Dist: {int(length)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

                if control_active and not muted:
                    vol = np.interp(length, [30, 200], [min_vol, max_vol])
                    vol_smooth = 0.9 * vol_smooth + 0.1 * vol
                    volume.SetMasterVolumeLevel(vol_smooth, None)
                    last_vol = vol_smooth

                    # Volume bar + color
                    vol_bar = np.interp(length, [30, 200], [400, 150])
                    vol_percent = np.interp(length, [30, 200], [0, 100])
                    color = (0, int(255 - vol_percent * 2.55), int(vol_percent * 2.55))
                    cv2.rectangle(img, (50, 150), (85, 400), color, 3)
                    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), color, cv2.FILLED)
                    cv2.putText(img, f'{int(vol_percent)} %', (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    # Lock and Mute Status
    lock_color = (0, 255, 0) if control_active else (0, 0, 255)
    mute_color = (0, 255, 255) if muted else (255, 255, 255)
    cv2.putText(img, f'Ctrl: {"ACTIVE" if control_active else "LOCKED"}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, lock_color, 2)
    cv2.putText(img, f'Muted: {"YES" if muted else "NO"}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mute_color, 2)

    # Display
    cv2.imshow("Advanced Volume Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
