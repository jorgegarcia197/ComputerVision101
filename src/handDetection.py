import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    success, img = webcam.read()
    
    # applying hand tracking model 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.Hands().process(img)
    
    # draw annotations on the image 
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawings.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam.release()
cv2.destroyAllWindows()