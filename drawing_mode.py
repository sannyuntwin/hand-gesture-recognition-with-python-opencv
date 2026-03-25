# ==============================
# ✋ HAND GESTURE DRAWING USING MEDIAPIPE & OPENCV
# ==============================
# This program detects your hand using your webcam,
# and when you point with one finger (index finger),
# you can draw on the screen by moving your finger.
# ==============================

# --- Import required libraries ---
import cv2               # OpenCV for camera input and image display
import mediapipe as mp    # MediaPipe for hand tracking and landmarks
import math               # Math functions for calculations
import numpy as np        # For array operations

# --- Initialize the MediaPipe Hands module ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Minimum confidence for hand detection
    min_tracking_confidence=0.7,   # Minimum confidence for landmark tracking
    max_num_hands=2                # Track up to 2 hands for dual drawing
)
mp_draw = mp.solutions.drawing_utils  # Utility to draw hand landmarks on the image

# --- Define the landmark indices for the fingertips ---
tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# --- Drawing canvas setup ---
canvas = None  # Will be initialized when we get first frame
left_drawing = False  # Whether left hand is drawing
right_drawing = False  # Whether right hand is drawing
left_last_point = None  # Last position of left hand drawing point
right_last_point = None  # Last position of right hand drawing point
left_color = (0, 255, 0)  # Green color for left hand
right_color = (255, 0, 0)  # Blue color for right hand
brush_size = 5  # Brush thickness

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
    thumb_tip_x = hand_landmarks.landmark[tip_ids[0]].x
    thumb_ip_x = hand_landmarks.landmark[tip_ids[0] - 1].x

    if handedness_label == 'Right':
        fingers.append(1 if thumb_tip_x < thumb_ip_x else 0)
    else:
        fingers.append(1 if thumb_tip_x > thumb_ip_x else 0)

    # --- Other 4 fingers ---
    for i in range(1, 5):
        tip_y = hand_landmarks.landmark[tip_ids[i]].y
        pip_y = hand_landmarks.landmark[tip_ids[i] - 2].y
        fingers.append(1 if tip_y < pip_y else 0)

    return fingers

# ==============================
# FUNCTION: Detect Pointing Gesture
# ==============================
def is_pointing(fingers):
    """
    Check if the hand is making a pointing gesture (only index finger up).
    Returns True if pointing, False otherwise.
    """
    return fingers[1] == 1 and sum(fingers) == 1

# ==============================
# MAIN LOOP: Capture from Webcam
# ==============================
cap = cv2.VideoCapture(0)  # Open the default webcam (0)

print("=== Hand Drawing Application (Two-Handed) ===")
print("Left Hand (Green): Point with index finger to draw")
print("Right Hand (Blue): Point with index finger to draw")
print("Both hands can draw simultaneously with different colors!")
print("Press 'c' to clear canvas")
print("Press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()  # Read one frame from the camera
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Initialize canvas on first frame
    if canvas is None:
        canvas = np.zeros_like(frame)  # Create black canvas with same dimensions as frame

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

            # Check if pointing (only index finger up)
            if is_pointing(finger_list):
                # Get index finger tip position
                index_finger_tip = hand_landmarks.landmark[8]  # Landmark 8 is index finger tip
                
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                x = int(index_finger_tip.x * w)
                y = int(index_finger_tip.y * h)
                
                # Draw a circle at the finger tip position with hand-specific color
                if handedness_label == 'Left':
                    color = left_color
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)  # Yellow circle for finger tip
                    
                    # Draw on canvas with left hand
                    if left_last_point is not None:
                        cv2.line(canvas, left_last_point, (x, y), left_color, brush_size)
                    left_last_point = (x, y)
                    left_drawing = True
                    
                    # Display status
                    cv2.putText(frame, "L: DRAWING", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, left_color, 2, cv2.LINE_AA)
                else:  # Right hand
                    color = right_color
                    cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)  # Cyan circle for finger tip
                    
                    # Draw on canvas with right hand
                    if right_last_point is not None:
                        cv2.line(canvas, right_last_point, (x, y), right_color, brush_size)
                    right_last_point = (x, y)
                    right_drawing = True
                    
                    # Display status
                    cv2.putText(frame, "R: DRAWING", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, right_color, 2, cv2.LINE_AA)
            else:
                # Not pointing, stop drawing for this hand
                if handedness_label == 'Left':
                    left_drawing = False
                    left_last_point = None
                else:  # Right hand
                    right_drawing = False
                    right_last_point = None
                
                # Display finger count
                num_fingers = finger_list.count(1)
                if handedness_label == 'Left':
                    cv2.putText(frame, f"L: {num_fingers} fingers", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"R: {num_fingers} fingers", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    else:
        # No hand detected
        left_drawing = False
        right_drawing = False
        left_last_point = None
        right_last_point = None

    # Combine frame with canvas (show drawing on top of camera feed)
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    
    # Show the webcam video with annotations and drawing
    cv2.imshow('Hand Drawing', combined)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break  # Press 'q' to quit
    elif key == ord('c'):
        # Clear canvas
        canvas = np.zeros_like(frame)
        print("Canvas cleared!")

# --- Cleanup ---
cap.release()           # Stop the webcam
cv2.destroyAllWindows() # Close all OpenCV windows
