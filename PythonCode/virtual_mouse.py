import cv2
import mediapipe as mp
import pyautogui
import time

# Webcam size
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# MediaPipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Cursor smoothing
prev_x, prev_y = 0, 0
smoothening = 7

# FPS
prev_time = 0

# Scroll tracking
prev_scroll_y = 0

# Finger state checker
def fingers_up(lm_list):
    fingers = []
    # Thumb
    fingers.append(lm_list[4][0] < lm_list[3][0])
    # Index
    fingers.append(lm_list[8][1] < lm_list[6][1])
    # Middle
    fingers.append(lm_list[12][1] < lm_list[10][1])
    # Ring
    fingers.append(lm_list[16][1] < lm_list[14][1])
    # Pinky
    fingers.append(lm_list[20][1] < lm_list[18][1])
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * wCam), int(lm.y * hCam)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if lm_list:
                fingers = fingers_up(lm_list)

                # Index finger tip
                x1, y1 = lm_list[8]
                # Thumb tip
                x2, y2 = lm_list[4]

                # === SCROLL MODE ===
                if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                    cv2.putText(img, "Mode: Scroll", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    current_scroll_y = lm_list[8][1]
                    scroll_delta = current_scroll_y - prev_scroll_y

                    if abs(scroll_delta) > 5:
                        pyautogui.scroll(-int(scroll_delta))

                    prev_scroll_y = current_scroll_y

                # === MOUSE MOVE + CLICK MODE ===
                elif fingers[1] and not fingers[2]:
                    # Map to screen
                    target_x = int(x1 * screen_width / wCam)
                    target_y = int(y1 * screen_height / hCam)

                    # Smooth movement
                    curr_x = prev_x + (target_x - prev_x) // smoothening
                    curr_y = prev_y + (target_y - prev_y) // smoothening

                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                    # Show cursor dot
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

                    # Click if thumb close to index
                    distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
                    if distance < 40:
                        pyautogui.click()
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
