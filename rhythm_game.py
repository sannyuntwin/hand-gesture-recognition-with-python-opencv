# ==============================
# 🎵 HAND RHYTHM GAME
# ==============================
# This game tests your rhythm and timing with hand gestures!
# Follow the gesture patterns that appear on screen in sync with the beat.
# Different gestures correspond to different actions:
# - Fist (0 fingers) = Red beat
# - Point (1 finger) = Blue beat  
# - Peace (2 fingers) = Green beat
# - Open Palm (5 fingers) = Yellow beat
# ==============================

# --- Import required libraries ---
import cv2               # OpenCV for camera input and image display
import mediapipe as mp    # MediaPipe for hand tracking and landmarks
import numpy as np        # For array operations
import math               # Math functions for calculations
import random            # For random patterns
import time              # For timing and rhythm

# --- Initialize the MediaPipe Hands module ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Minimum confidence for hand detection
    min_tracking_confidence=0.7,   # Minimum confidence for landmark tracking
    max_num_hands=2                # Track up to 2 hands for dual gameplay
)
mp_draw = mp.solutions.drawing_utils  # Utility to draw hand landmarks on the image

# --- Define the landmark indices for the fingertips ---
tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# --- Game parameters ---
SCREEN_WIDTH = 800  # Wider for two-handed play
SCREEN_HEIGHT = 480
FPS = 30
BEAT_INTERVAL = 0.8  # Slightly faster for two hands
BEAT_SPEED = 5.0     # How fast beats move down
HIT_ZONE_Y = 400     # Y position where beats should be hit
HIT_ZONE_HEIGHT = 50 # Height of the hit zone

# Two-handed lane configuration
LEFT_HAND_LANES = [200, 300]  # Left hand beats
RIGHT_HAND_LANES = [500, 600] # Right hand beats

# --- Gesture colors ---
GESTURE_COLORS = {
    0: (0, 0, 255),    # Red - Fist
    1: (255, 0, 0),    # Blue - Point
    2: (0, 255, 0),    # Green - Peace
    5: (0, 255, 255)   # Yellow - Open Palm
}

GESTURE_NAMES = {
    0: "FIST",
    1: "POINT", 
    2: "PEACE",
    5: "PALM"
}

# --- Game state ---
class GameState:
    def __init__(self):
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.beats = []  # List of active beats
        self.last_beat_time = time.time()
        self.game_started = False
        self.game_over = False
        self.misses = 0
        self.perfect_hits = 0
        self.good_hits = 0
        self.left_gesture = None   # Left hand gesture
        self.right_gesture = None  # Right hand gesture
        self.left_hand_detected = False
        self.right_hand_detected = False

# --- Beat class ---
class Beat:
    def __init__(self, gesture_type, lane_x, hand_side):
        self.gesture_type = gesture_type
        self.lane_x = lane_x
        self.hand_side = hand_side  # 'left' or 'right'
        self.y = 50  # Start from top
        self.hit = False
        self.missed = False
        
    def update(self, dt):
        self.y += BEAT_SPEED * dt * 60  # Move down
        
        # Check if beat passed the hit zone
        if self.y > HIT_ZONE_Y + HIT_ZONE_HEIGHT and not self.hit and not self.missed:
            self.missed = True
            return True  # Return True when beat is missed
        return False
    
    def draw(self, frame):
        if not self.hit and not self.missed:
            color = GESTURE_COLORS.get(self.gesture_type, (255, 255, 255))
            # Draw beat circle
            cv2.circle(frame, (self.lane_x, int(self.y)), 30, color, -1)
            # Draw gesture text
            text = GESTURE_NAMES.get(self.gesture_type, "?")
            cv2.putText(frame, text, (self.lane_x - 25, int(self.y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
# FUNCTION: Get Gesture Type
# ==============================
def get_gesture_type(fingers):
    """
    Convert finger list to gesture type.
    Returns the number of fingers up for specific gestures.
    """
    num_fingers = sum(fingers)
    
    # Only return specific gesture types we use in the game
    if num_fingers == 0:
        return 0  # Fist
    elif num_fingers == 1 and fingers[1] == 1:
        return 1  # Point (index finger only)
    elif num_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
        return 2  # Peace (index + middle)
    elif num_fingers == 5:
        return 5  # Open palm
    
    return None  # Invalid gesture for game

# ==============================
# FUNCTION: Check Hit
# ==============================
def check_hit(beat, left_gesture, right_gesture):
    """
    Check if the current gesture hits the beat.
    """
    if beat.hit or beat.missed:
        return False
    
    # Check if beat is in hit zone
    if HIT_ZONE_Y - HIT_ZONE_HEIGHT <= beat.y <= HIT_ZONE_Y + HIT_ZONE_HEIGHT:
        # Check if gesture matches for the correct hand
        current_gesture = left_gesture if beat.hand_side == 'left' else right_gesture
        
        if current_gesture == beat.gesture_type:
            # Calculate timing accuracy
            distance_from_center = abs(beat.y - HIT_ZONE_Y)
            if distance_from_center < 10:
                return "PERFECT"
            elif distance_from_center < 25:
                return "GOOD"
    
    return False

# ==============================
# FUNCTION: Draw Game UI
# ==============================
def draw_game_ui(frame, game_state):
    """
    Draw the game interface elements.
    """
    # Draw lanes for both hands
    all_lanes = LEFT_HAND_LANES + RIGHT_HAND_LANES
    
    # Draw left hand lanes (blue tint)
    for x in LEFT_HAND_LANES:
        cv2.line(frame, (x, 50), (x, SCREEN_HEIGHT - 50), (100, 100, 255), 2)
    
    # Draw right hand lanes (green tint)
    for x in RIGHT_HAND_LANES:
        cv2.line(frame, (x, 50), (x, SCREEN_HEIGHT - 50), (100, 255, 100), 2)
    
    # Draw center divider
    cv2.line(frame, (SCREEN_WIDTH//2, 50), (SCREEN_WIDTH//2, SCREEN_HEIGHT - 50), (200, 200, 200), 3)
    
    # Draw hand labels
    cv2.putText(frame, "LEFT HAND", (150, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT HAND", (520, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
    
    # Draw hit zone
    cv2.rectangle(frame, (100, HIT_ZONE_Y - HIT_ZONE_HEIGHT), 
                 (SCREEN_WIDTH - 100, HIT_ZONE_Y + HIT_ZONE_HEIGHT), 
                 (255, 255, 255), 2)
    
    # Draw score
    cv2.putText(frame, f"Score: {game_state.score}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw combo
    if game_state.combo > 0:
        cv2.putText(frame, f"Combo: {game_state.combo}x", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Draw misses counter
    cv2.putText(frame, f"Misses: {game_state.misses}", (SCREEN_WIDTH - 150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Draw current gesture indicators for both hands
    if game_state.left_gesture is not None:
        color = GESTURE_COLORS.get(game_state.left_gesture, (255, 255, 255))
        text = GESTURE_NAMES.get(game_state.left_gesture, "?")
        cv2.putText(frame, f"L: {text}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    if game_state.right_gesture is not None:
        color = GESTURE_COLORS.get(game_state.right_gesture, (255, 255, 255))
        text = GESTURE_NAMES.get(game_state.right_gesture, "?")
        cv2.putText(frame, f"R: {text}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    # Draw hand detection status
    left_status = "✓" if game_state.left_hand_detected else "✗"
    right_status = "✓" if game_state.right_hand_detected else "✗"
    cv2.putText(frame, f"L:{left_status} R:{right_status}", (SCREEN_WIDTH - 100, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# ==============================
# MAIN GAME LOOP
# ==============================
def main():
    # Initialize game state
    game = GameState()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    
    print("=== Hand Rhythm Game (Two-Handed) ===")
    print("Instructions:")
    print("  • Use BOTH hands to hit beats in their respective lanes")
    print("  • LEFT HAND: Blue lanes on the left side")
    print("  • RIGHT HAND: Green lanes on the right side")
    print("  • FIST (0 fingers) for red beats")
    print("  • POINT (1 finger) for blue beats") 
    print("  • PEACE (2 fingers) for green beats")
    print("  • PALM (5 fingers) for yellow beats")
    print("  • Press SPACE to start/pause")
    print("  • Press 'q' to quit")
    
    last_time = time.time()
    
    while cap.isOpened():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        success, frame = cap.read()
        if not success:
            continue
        
        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Process hand detection
        game.left_hand_detected = False
        game.right_hand_detected = False
        game.left_gesture = None
        game.right_gesture = None
        
        if results.multi_hand_landmarks:
            for i, (hand_landmarks, hand_handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Count fingers and get gesture
                finger_list = count_fingers(hand_landmarks, hand_handedness.classification[0].label)
                gesture = get_gesture_type(finger_list)
                
                # Determine which hand this is
                handedness = hand_handedness.classification[0].label
                
                if handedness == 'Left':
                    game.left_hand_detected = True
                    game.left_gesture = gesture
                else:  # Right hand
                    game.right_hand_detected = True
                    game.right_gesture = gesture
        
        # Generate new beats
        if game.game_started and not game.game_over:
            if current_time - game.last_beat_time > BEAT_INTERVAL:
                # Create random beat for either hand
                gesture = random.choice([0, 1, 2, 5])  # Random gesture type
                
                # Randomly choose hand and lane
                if random.choice([True, False]):  # Left hand
                    lane = random.choice(LEFT_HAND_LANES)
                    hand_side = 'left'
                else:  # Right hand
                    lane = random.choice(RIGHT_HAND_LANES)
                    hand_side = 'right'
                
                game.beats.append(Beat(gesture, lane, hand_side))
                game.last_beat_time = current_time
        
        # Update beats
        if game.game_started and not game.game_over:
            for beat in game.beats[:]:  # Copy list to avoid modification issues
                if beat.update(dt):
                    # Beat was missed
                    game.misses += 1
                    game.combo = 0
                    game.beats.remove(beat)
                    
                    # Game over if too many misses
                    if game.misses >= 15:  # More misses for two-handed
                        game.game_over = True
                
                # Check for hits with both hands
                hit_result = check_hit(beat, game.left_gesture, game.right_gesture)
                if hit_result:
                    beat.hit = True
                    game.beats.remove(beat)
                    
                    if hit_result == "PERFECT":
                        game.score += 100
                        game.perfect_hits += 1
                    else:  # GOOD
                        game.score += 50
                        game.good_hits += 1
                    
                    game.combo += 1
                    if game.combo > game.max_combo:
                        game.max_combo = game.combo
        
        # Draw everything
        draw_game_ui(frame, game)
        
        # Draw beats
        for beat in game.beats:
            beat.draw(frame)
        
        # Draw game over screen
        if game.game_over:
            cv2.putText(frame, "GAME OVER", (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Final Score: {game.score}", (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Max Combo: {game.max_combo}x", (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press SPACE to restart", (SCREEN_WIDTH//2 - 140, SCREEN_HEIGHT//2 + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw start screen
        if not game.game_started and not game.game_over:
            cv2.putText(frame, "HAND RHYTHM GAME", (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Press SPACE to start", (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow('Hand Rhythm Game', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            if game.game_over:
                # Reset game
                game = GameState()
            else:
                # Toggle game start/pause
                game.game_started = not game.game_started
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
