# ==============================
# ✋ 3D CUBE CONTROL WITH HAND GESTURES
# ==============================
# This program detects your hand using your webcam,
# and allows you to control a 3D cube with hand gestures:
# - Move cube with hand position
# - Rotate with finger gestures
# ==============================

# --- Import required libraries ---
import cv2               # OpenCV for camera input and image display
import mediapipe as mp    # MediaPipe for hand tracking and landmarks
import numpy as np        # For array operations
import math               # Math functions for calculations

# --- Initialize the MediaPipe Hands module ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Minimum confidence for hand detection
    min_tracking_confidence=0.7,   # Minimum confidence for landmark tracking
    max_num_hands=2                # Track up to 2 hands for enhanced control
)
mp_draw = mp.solutions.drawing_utils  # Utility to draw hand landmarks on the image

# --- Define the landmark indices for the fingertips ---
tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# --- 3D Cube parameters ---
cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
], dtype=float)

cube_edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Back face edges
    [4, 5], [5, 6], [6, 7], [7, 4],  # Front face edges
    [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
]

# --- Control parameters ---
cube_position = np.array([0.0, 0.0, 0.0])  # x, y, z position
cube_rotation = np.array([0.0, 0.0, 0.0])  # rotation angles (x, y, z)
cube_scale = 100.0  # Scale factor for display
cube_color = (0, 255, 0)  # Green color for cube edges
auto_rotate = False  # Auto-rotation toggle
left_hand_center_prev = None  # Previous left hand center for movement tracking
right_hand_center_prev = None  # Previous right hand center for scaling

# ==============================
# FUNCTION: Count Fingers
# ==============================
def count_fingers(hand_landmarks, handedness_label):
    """
    Determine which fingers are open (1) or closed (0)
    for a given hand.
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
# FUNCTION: Get Hand Center
# ==============================
def get_hand_center(hand_landmarks):
    """
    Calculate the center point of the hand.
    """
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
    z_coords = [landmark.z for landmark in hand_landmarks.landmark]
    
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    center_z = sum(z_coords) / len(z_coords)
    
    return np.array([center_x, center_y, center_z])

# ==============================
# FUNCTION: Rotate Point
# ==============================
def rotate_point(point, angles):
    """
    Rotate a 3D point by given angles (in radians).
    """
    # Rotation around X axis
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    
    # Rotation around Y axis
    ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    
    # Rotation around Z axis
    rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    # Apply rotations
    rotated = np.dot(rx, point)
    rotated = np.dot(ry, rotated)
    rotated = np.dot(rz, rotated)
    
    return rotated

# ==============================
# FUNCTION: Project 3D to 2D
# ==============================
def project_3d_to_2d(point_3d, width, height):
    """
    Project a 3D point to 2D screen coordinates.
    """
    # Simple orthographic projection
    x_2d = int(point_3d[0] + width / 2)
    y_2d = int(point_3d[1] + height / 2)
    
    return (x_2d, y_2d)

# ==============================
# FUNCTION: Draw 3D Cube
# ==============================
def draw_cube(frame, position, rotation, scale, color):
    """
    Draw a 3D cube on the frame.
    """
    height, width = frame.shape[:2]
    
    # Transform vertices
    transformed_vertices = []
    for vertex in cube_vertices:
        # Apply rotation
        rotated = rotate_point(vertex * scale, rotation)
        # Apply position
        transformed = rotated + position + np.array([width/2, height/2, 0])
        transformed_vertices.append(transformed)
    
    transformed_vertices = np.array(transformed_vertices)
    
    # Project to 2D and draw edges
    projected_vertices = []
    for vertex in transformed_vertices:
        x_2d = int(vertex[0])
        y_2d = int(vertex[1])
        projected_vertices.append((x_2d, y_2d))
    
    # Draw edges with dynamic color
    for edge in cube_edges:
        pt1 = projected_vertices[edge[0]]
        pt2 = projected_vertices[edge[1]]
        cv2.line(frame, pt1, pt2, color, 2)
    
    # Draw vertices
    for vertex in projected_vertices:
        cv2.circle(frame, vertex, 4, (0, 0, 255), -1)

# ==============================
# MAIN LOOP: Capture from Webcam
# ==============================
cap = cv2.VideoCapture(0)  # Open the default webcam (0)

print("=== 3D Cube Control (Two-Handed) ===")
print("Left Hand Controls:")
print("  • Open palm (5 fingers) - Move cube with left hand position")
print("  • Point (1 finger) - Rotate around Y axis")
print("  • Peace (2 fingers) - Rotate around X axis")
print("  • Fist (0 fingers) - Rotate around Z axis")
print("Right Hand Controls:")
print("  • Open palm (5 fingers) - Scale cube up/down")
print("  • Point (1 finger) - Reset position")
print("  • Peace (2 fingers) - Change cube color")
print("  • Fist (0 fingers) - Auto-rotate toggle")
print("Press 'r' to reset cube")
print("Press 'q' to quit")

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

    # If any hands are detected, analyze them
    if results.multi_hand_landmarks:
        # Loop through all detected hands (if more than one)
        for i, (hand_landmarks, hand_handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            # Draw hand landmarks and connections on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get whether it's Left or Right hand
            handedness_label = hand_handedness.classification[0].label

            # Count open fingers
            finger_list = count_fingers(hand_landmarks, handedness_label)
            num_fingers = finger_list.count(1)

            # Get hand center position
            hand_center = get_hand_center(hand_landmarks)
            h, w, _ = frame.shape
            hand_center_screen = np.array([hand_center[0] * w, hand_center[1] * h])

            # Control cube based on hand and finger count
            if handedness_label == 'Left':
                # LEFT HAND CONTROLS
                if num_fingers == 5:  # Open palm - move cube
                    if left_hand_center_prev is not None:
                        movement = hand_center_screen - left_hand_center_prev
                        cube_position[0] += movement[0] * 0.5
                        cube_position[1] += movement[1] * 0.5
                    left_hand_center_prev = hand_center_screen.copy()
                    cv2.putText(frame, "L: MOVE", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                elif num_fingers == 1:  # Point - rotate Y axis
                    cube_rotation[1] += 0.05
                    left_hand_center_prev = None
                    cv2.putText(frame, "L: ROTATE Y", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    
                elif num_fingers == 2:  # Peace - rotate X axis
                    cube_rotation[0] += 0.05
                    left_hand_center_prev = None
                    cv2.putText(frame, "L: ROTATE X", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    
                elif num_fingers == 0:  # Fist - rotate Z axis
                    cube_rotation[2] += 0.05
                    left_hand_center_prev = None
                    cv2.putText(frame, "L: ROTATE Z", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                else:
                    left_hand_center_prev = None
                    
            else:  # RIGHT HAND CONTROLS
                if num_fingers == 5:  # Open palm - scale cube
                    if right_hand_center_prev is not None:
                        movement = hand_center_screen[1] - right_hand_center_prev[1]
                        cube_scale += movement * 0.5
                        cube_scale = max(20, min(200, cube_scale))  # Limit scale
                    right_hand_center_prev = hand_center_screen.copy()
                    cv2.putText(frame, "R: SCALE", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    
                elif num_fingers == 1:  # Point - reset position
                    cube_position = np.array([0.0, 0.0, 0.0])
                    right_hand_center_prev = None
                    cv2.putText(frame, "R: RESET", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                    
                elif num_fingers == 2:  # Peace - change color
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    current_color_index = colors.index(cube_color) if cube_color in colors else 0
                    cube_color = colors[(current_color_index + 1) % len(colors)]
                    right_hand_center_prev = None
                    cv2.putText(frame, "R: COLOR", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                    
                elif num_fingers == 0:  # Fist - auto-rotate toggle
                    auto_rotate = not auto_rotate
                    right_hand_center_prev = None
                    cv2.putText(frame, "R: AUTO-ROTATE", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                else:
                    right_hand_center_prev = None
    else:
        left_hand_center_prev = None
        right_hand_center_prev = None

    # Auto-rotate if enabled
    if auto_rotate:
        cube_rotation[0] += 0.01
        cube_rotation[1] += 0.02
        cube_rotation[2] += 0.005
        cv2.putText(frame, "AUTO-ROTATE ON", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw the 3D cube
    draw_cube(frame, cube_position, cube_rotation, cube_scale, cube_color)

    # Show the webcam video with cube
    cv2.imshow('3D Cube Control', frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break  # Press 'q' to quit
    elif key == ord('r'):
        # Reset cube position and rotation
        cube_position = np.array([0.0, 0.0, 0.0])
        cube_rotation = np.array([0.0, 0.0, 0.0])
        print("Cube reset!")

# --- Cleanup ---
cap.release()           # Stop the webcam
cv2.destroyAllWindows() # Close all OpenCV windows
