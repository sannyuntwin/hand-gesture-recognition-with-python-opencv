# ==============================
# ✋ HAND GESTURE RECOGNITION USING MEDIAPIPE & OPENCV
# ==============================
# This program detects your hand using your webcam,
# counts how many fingers you show, and recognizes simple gestures
# like Thumbs Up, Peace, Rock, or Call Me.
# ==============================

# --- Import required libraries ---
import cv2               # OpenCV for camera input and image display
import mediapipe as mp    # MediaPipe for hand tracking and landmarks
import math               # Math functions for calculations

# --- Initialize the MediaPipe Hands module ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Minimum confidence for hand detection
    min_tracking_confidence=0.7,   # Minimum confidence for landmark tracking
    max_num_hands=2                # Detect up to 2 hands
)
mp_draw = mp.solutions.drawing_utils  # Utility to draw hand landmarks on the image

# --- Define the landmark indices for the fingertips ---
# Each finger has a specific landmark index in MediaPipe's hand model.
tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# ==============================
# FUNCTION: Count Fingers
# ==============================
def count_fingers(hand_landmarks, handedness_label):
    """
    Determine which fingers are open (1) or closed (0)
    for a given hand.
    - hand_landmarks: the 21 points detected on the hand
    - handedness_label: 'Left' or 'Right' (helps detect thumb correctly)
    Returns: [thumb, index, middle, ring, pinky] → each 1 or 0
    """
    fingers = []

    # --- Thumb detection ---
    # For thumb, we compare X positions instead of Y,
    # because the thumb moves sideways, not vertically.
    thumb_tip_x = hand_landmarks.landmark[tip_ids[0]].x
    thumb_ip_x = hand_landmarks.landmark[tip_ids[0] - 1].x  # Landmark before thumb tip (joint)

    # The direction of thumb differs for left and right hand, so we check handedness.
    if handedness_label == 'Right':
        fingers.append(1 if thumb_tip_x < thumb_ip_x else 0)
    else:
        fingers.append(1 if thumb_tip_x > thumb_ip_x else 0)

    # --- Other 4 fingers ---
    # For each finger, if the tip is above (smaller y value) the knuckle joint, the finger is open.
    for i in range(1, 5):
        tip_y = hand_landmarks.landmark[tip_ids[i]].y
        pip_y = hand_landmarks.landmark[tip_ids[i] - 2].y  # Joint below the tip
        fingers.append(1 if tip_y < pip_y else 0)

    return fingers


# ==============================
# FUNCTION: Detect Gesture
# ==============================
def detect_gesture(fingers, hand_landmarks):
    """
    Detects which hand gesture is being made based on finger positions.
    Returns a string label of the gesture.
    """
    num = sum(fingers)  # Count how many fingers are open
    wrist_y = hand_landmarks.landmark[0].y
    thumb_tip_y = hand_landmarks.landmark[4].y

    # --- Gesture Rules ---
    # Fist (no fingers open)
    if num == 0:
        return "Fist"

    # Palm (all fingers open)
    if num == 5:
        return "Palm"

    # Pointing (only index finger open)
    if fingers[1] == 1 and sum(fingers) == 1:
        return "Pointing"

    # Thumbs Up / Down (only thumb open)
    if fingers[0] == 1 and sum(fingers[1:]) == 0:
        # Compare thumb tip height relative to wrist
        if thumb_tip_y < wrist_y - 0.02:
            return "Thumbs Up"
        elif thumb_tip_y > wrist_y + 0.02:
            return "Thumbs Down"
        else:
            return "Thumb (Horizontal)"

    # Peace ✌️ (index + middle fingers open)
    if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
        return "Peace"

    # Rock 🤘 (thumb, index, pinky open)
    if fingers[0] == 1 and fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0:
        return "Rock"

    # Call Me 🤙 (thumb + pinky open)
    if fingers[0] == 1 and fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
        return "Call Me"

    # Otherwise, just show how many fingers are open
    return f"{num} Fingers"


# ==============================
# MAIN LOOP: Capture from Webcam
# ==============================
cap = cv2.VideoCapture(0)  # Open the default webcam (0)

while cap.isOpened():
    success, frame = cap.read()  # Read one frame from the camera
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a mirror view (like a selfie)
    frame = cv2.flip(frame, 1)

    # Convert image color from BGR (OpenCV default) to RGB (MediaPipe requires this)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands and landmarks
    results = hands.process(rgb_frame)

    # If any hands are detected, draw and analyze them
    if results.multi_hand_landmarks:
        # Loop through all detected hands (if more than one)
        for i, (hand_landmarks, hand_handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            # Draw hand landmarks and connections on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get whether it's Left or Right hand
            handedness_label = hand_handedness.classification[0].label  # 'Left' or 'Right'

            # Count open fingers
            finger_list = count_fingers(hand_landmarks, handedness_label)
            num_fingers = finger_list.count(1)

            # Detect gesture type
            gesture = detect_gesture(finger_list, hand_landmarks)

            # --- Display results on the screen ---
            x_pos = 10
            y_pos = 50 + i * 60  # Offset text for each hand detected
            cv2.putText(frame, f"Count: {num_fingers}", (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Gesture: {gesture}", (x_pos, y_pos + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2, cv2.LINE_AA)

    # Show the webcam video with annotations
    cv2.imshow('Hand Gesture Recognition', frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()           # Stop the webcam
cv2.destroyAllWindows() # Close all OpenCV windows
