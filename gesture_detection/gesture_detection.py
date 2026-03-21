
import cv2
import numpy as np
import time

print("Gesture Detection Module Started!")
print("Show your hand to the camera")
print("Press 'q' to return to menu")

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    fg_mask = bg_subtractor.apply(frame)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hand_contours = [c for c in contours if 5000 < cv2.contourArea(c) < 50000]
    
    if hand_contours:
        largest = max(hand_contours, key=cv2.contourArea)
        cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
        cv2.putText(frame, "Hand Detected!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
