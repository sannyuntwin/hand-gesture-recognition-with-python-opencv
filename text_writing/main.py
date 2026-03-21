# ==============================
# ✋ HAND GESTURE TEXT WRITING SYSTEM
# ==============================
# Write text using hand gestures - perfect for accessibility!
# ==============================

import cv2
import numpy as np
import time
import pyautogui
from collections import deque
import json
import math
import threading
import queue

# Configuration
CONFIG = {
    'smoothing_window': 5,
    'gesture_hold_time': 1.0,
    'mouse_sensitivity': 1.5,
    'save_gestures': True,
    'camera_index': 0,
    'auto_type_delay': 0.5
}

# Virtual keyboard layout
KEYBOARD_LAYOUT = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['SPACE', 'BACKSPACE', 'ENTER', 'CLEAR', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
]

class GestureTextWriter:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(CONFIG['camera_index'])
        
        # Background subtractor for hand detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Tracking variables
        self.gesture_history = deque(maxlen=CONFIG['smoothing_window'])
        self.last_gesture_time = time.time()
        self.current_gesture = "None"
        self.mouse_position = (0, 0)
        self.prev_mouse_pos = (0, 0)
        self.is_dragging = False
        
        # Text writing variables
        self.is_text_mode = False
        self.keyboard_pos = [0, 0]
        self.current_text = ""
        self.last_selected_char = None
        self.text_queue = queue.Queue()
        
        # Gesture counters
        self.gesture_stats = {}
        self.session_start = time.time()
        
        # Hand detection parameters
        self.hand_contour_min_area = 5000
        self.hand_contour_max_area = 50000
        
        # Start text typing thread
        self.typing_thread = threading.Thread(target=self.type_text_worker, daemon=True)
        self.typing_thread.start()
        
    def type_text_worker(self):
        """Worker thread for typing text"""
        while True:
            try:
                text_to_type = self.text_queue.get(timeout=1)
                if text_to_type == 'BACKSPACE':
                    pyautogui.press('backspace')
                elif text_to_type == 'ENTER':
                    pyautogui.press('enter')
                elif text_to_type == 'SPACE':
                    pyautogui.press('space')
                elif text_to_type == 'CLEAR':
                    # Select all and delete
                    pyautogui.hotkey('ctrl', 'a')
                    time.sleep(0.1)
                    pyautogui.press('backspace')
                elif len(text_to_type) == 1:
                    pyautogui.typewrite(text_to_type)
                time.sleep(CONFIG['auto_type_delay'])
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Typing error: {e}")
    
    def detect_hand(self, frame):
        """Detect hand using background subtraction and contour detection"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        hand_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.hand_contour_min_area < area < self.hand_contour_max_area:
                hand_contours.append(contour)
        
        return hand_contours, fg_mask
    
    def count_fingers_from_contour(self, contour, frame):
        """Count fingers from hand contour using convexity defects"""
        if len(contour) < 5:
            return 0, contour
        
        # Find convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # Find convexity defects
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is None:
            return 0, contour
        
        # Count fingers based on defects
        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate angle
            angle = self.calculate_angle(start, far, end)
            
            # Count as finger if angle is less than 90 degrees and depth is significant
            if angle < 90 and d > 10000:
                finger_count += 1
        
        # Adjust finger count (add 1 for the thumb if detected)
        if finger_count > 0:
            finger_count = min(finger_count + 1, 5)
        
        return finger_count, contour
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def detect_gesture_from_fingers(self, finger_count, contour_center):
        """Detect gesture based on finger count and position"""
        if finger_count == 0:
            return "Fist"
        elif finger_count == 5:
            return "Palm"
        elif finger_count == 1:
            return "Point"
        elif finger_count == 2:
            return "Peace"
        elif finger_count == 3:
            return "Three_Fingers"
        else:
            return f"{finger_count}_Fingers"
    
    def navigate_keyboard(self, direction):
        """Navigate virtual keyboard"""
        if direction == "right":
            self.keyboard_pos[1] += 1
            if self.keyboard_pos[1] >= len(KEYBOARD_LAYOUT[self.keyboard_pos[0]]):
                self.keyboard_pos[1] = 0
        elif direction == "left":
            self.keyboard_pos[1] -= 1
            if self.keyboard_pos[1] < 0:
                self.keyboard_pos[1] = len(KEYBOARD_LAYOUT[self.keyboard_pos[0]]) - 1
        elif direction == "down":
            self.keyboard_pos[0] += 1
            if self.keyboard_pos[0] >= len(KEYBOARD_LAYOUT):
                self.keyboard_pos[0] = 0
            elif self.keyboard_pos[1] >= len(KEYBOARD_LAYOUT[self.keyboard_pos[0]]):
                self.keyboard_pos[1] = len(KEYBOARD_LAYOUT[self.keyboard_pos[0]]) - 1
        elif direction == "up":
            self.keyboard_pos[0] -= 1
            if self.keyboard_pos[0] < 0:
                self.keyboard_pos[0] = len(KEYBOARD_LAYOUT) - 1
            elif self.keyboard_pos[1] >= len(KEYBOARD_LAYOUT[self.keyboard_pos[0]]):
                self.keyboard_pos[1] = len(KEYBOARD_LAYOUT[self.keyboard_pos[0]]) - 1
    
    def get_current_key(self):
        """Get current key from keyboard position"""
        if self.keyboard_pos[0] < len(KEYBOARD_LAYOUT):
            row = KEYBOARD_LAYOUT[self.keyboard_pos[0]]
            if self.keyboard_pos[1] < len(row):
                return row[self.keyboard_pos[1]]
        return None
    
    def execute_gesture_action(self, gesture, contour_center=None):
        """Execute actions based on detected gestures"""
        screen_width, screen_height = pyautogui.size()
        
        if self.is_text_mode:
            # Text writing mode actions
            if gesture == "Point" and contour_center:
                # Navigate keyboard based on hand position
                x, y = contour_center
                frame_center_x, frame_center_y = 320, 240
                
                if x > frame_center_x + 50:
                    self.navigate_keyboard("right")
                elif x < frame_center_x - 50:
                    self.navigate_keyboard("left")
                elif y > frame_center_y + 50:
                    self.navigate_keyboard("down")
                elif y < frame_center_y - 50:
                    self.navigate_keyboard("up")
                    
            elif gesture == "Fist":
                # Select current character
                current_key = self.get_current_key()
                if current_key:
                    self.text_queue.put(current_key)
                    self.last_selected_char = current_key
                    print(f"Typed: {current_key}")
                    
            elif gesture == "Three_Fingers":
                # Type space
                self.text_queue.put('SPACE')
                print("Typed: SPACE")
                
            elif gesture == "Two_Fingers":
                # Backspace
                self.text_queue.put('BACKSPACE')
                print("Typed: BACKSPACE")
                
            elif gesture == "Call_Me":
                # Enter
                self.text_queue.put('ENTER')
                print("Typed: ENTER")
                
            elif gesture == "Rock":
                # Clear text
                self.text_queue.put('CLEAR')
                print("Typed: CLEAR")
                
            elif gesture == "Palm":
                # Switch to mouse mode
                self.is_text_mode = False
                print("Switched to MOUSE MODE")
                
        else:
            # Mouse control mode actions
            if gesture == "Point" and contour_center:
                # Virtual mouse control
                mouse_x = int((1 - contour_center[0] / 640) * screen_width)
                mouse_y = int((contour_center[1] / 480) * screen_height)
                
                # Smooth movement
                if self.prev_mouse_pos != (0, 0):
                    smooth_x = int(self.prev_mouse_pos[0] + (mouse_x - self.prev_mouse_pos[0]) * 0.3)
                    smooth_y = int(self.prev_mouse_pos[1] + (mouse_y - self.prev_mouse_pos[1]) * 0.3)
                    pyautogui.moveTo(smooth_x, smooth_y)
                else:
                    pyautogui.moveTo(mouse_x, mouse_y)
                
                self.prev_mouse_pos = (mouse_x, mouse_y)
                
            elif gesture == "Fist":
                # Click action
                if not self.is_dragging:
                    pyautogui.click()
                    self.is_dragging = True
                    time.sleep(0.2)
                else:
                    self.is_dragging = False
                
            elif gesture == "Peace":
                # Next slide/presentation
                pyautogui.press('right')
                
            elif gesture == "Palm":
                # Switch to text mode
                self.is_text_mode = True
                self.keyboard_pos = [0, 0]
                print("Switched to TEXT MODE")
                print("Open any text editor (Notepad, Word, etc.) to start typing!")
            
            elif gesture == "Three_Fingers":
                # Volume up
                print("Volume up gesture detected")
                
            elif gesture == "Two_Fingers":
                # Volume down
                print("Volume down gesture detected")
    
    def update_gesture_stats(self, gesture):
        """Update gesture statistics"""
        if gesture not in self.gesture_stats:
            self.gesture_stats[gesture] = 0
        self.gesture_stats[gesture] += 1
    
    def draw_enhanced_ui(self, frame, gesture, contours=None):
        """Draw enhanced UI with gesture information and controls"""
        h, w, _ = frame.shape
        
        # Draw mode indicator
        mode_text = "TEXT MODE" if self.is_text_mode else "MOUSE MODE"
        mode_color = (0, 255, 0) if self.is_text_mode else (255, 0, 0)
        cv2.putText(frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        
        # Draw gesture info
        cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw hand contours
        if contours:
            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
                
                # Draw contour center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        
        # Draw control instructions based on mode
        if self.is_text_mode:
            instructions = [
                "TEXT MODE:",
                "Point: Navigate keyboard",
                "Fist: Select character",
                "Three Fingers: SPACE",
                "Two Fingers: BACKSPACE",
                "Call Me: ENTER",
                "Rock: CLEAR",
                "Palm: Switch to Mouse"
            ]
            
            # Draw virtual keyboard
            self.draw_virtual_keyboard(frame)
            
            # Highlight current position
            current_key = self.get_current_key()
            if current_key:
                cv2.putText(frame, f"Selected: {current_key}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            instructions = [
                "MOUSE MODE:",
                "Point: Move cursor",
                "Fist: Click",
                "Peace: Next slide",
                "Palm: Switch to Text",
                "Three Fingers: Volume +",
                "Two Fingers: Volume -"
            ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, h - 140 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw statistics
        session_time = int(time.time() - self.session_start)
        cv2.putText(frame, f"Session: {session_time}s", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def draw_virtual_keyboard(self, frame):
        """Draw virtual keyboard on screen"""
        h, w = frame.shape
        keyboard_start_y = 150
        key_size = 30
        key_spacing = 35
        
        for row_idx, row in enumerate(KEYBOARD_LAYOUT):
            y_pos = keyboard_start_y + row_idx * key_spacing
            for col_idx, key in enumerate(row):
                x_pos = 10 + col_idx * key_spacing
                
                # Highlight current key
                if row_idx == self.keyboard_pos[0] and col_idx == self.keyboard_pos[1]:
                    color = (0, 255, 0)  # Green for selected
                    thickness = 3
                else:
                    color = (255, 255, 255)  # White for others
                    thickness = 1
                
                # Draw key rectangle
                cv2.rectangle(frame, (x_pos, y_pos), 
                            (x_pos + key_size, y_pos + key_size), 
                            color, thickness)
                
                # Draw key text
                font_scale = 0.4 if len(key) > 1 else 0.6
                cv2.putText(frame, key, (x_pos + 5, y_pos + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    
    def save_session_data(self):
        """Save session statistics to file"""
        if CONFIG['save_gestures']:
            session_data = {
                'duration': time.time() - self.session_start,
                'gesture_stats': self.gesture_stats,
                'timestamp': time.time(),
                'text_mode_used': self.is_text_mode
            }
            
            with open('gesture_session.json', 'w') as f:
                json.dump(session_data, f, indent=2)
    
    def run(self):
        """Main execution loop"""
        print("🖐️  Hand Gesture Text Writing System Started!")
        print("=" * 50)
        print("INSTRUCTIONS:")
        print("1. Start by showing PALM gesture to switch to TEXT MODE")
        print("2. Open any text editor (Notepad, Word, etc.)")
        print("3. Use POINT gesture to navigate the virtual keyboard")
        print("4. Use FIST gesture to select and type characters")
        print("5. Use other gestures for special keys:")
        print("   - Three Fingers: SPACE")
        print("   - Two Fingers: BACKSPACE")
        print("   - Call Me: ENTER")
        print("   - Rock: CLEAR all text")
        print("   - Palm: Switch back to MOUSE MODE")
        print("=" * 50)
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip for mirror view
                frame = cv2.flip(frame, 1)
                
                # Detect hand
                hand_contours, fg_mask = self.detect_hand(frame)
                
                current_gesture = "No Hand Detected"
                contour_center = None
                
                if hand_contours:
                    # Process the largest contour
                    largest_contour = max(hand_contours, key=cv2.contourArea)
                    
                    # Count fingers
                    finger_count, _ = self.count_fingers_from_contour(largest_contour, frame)
                    
                    # Detect gesture
                    current_gesture = self.detect_gesture_from_fingers(finger_count, None)
                    
                    # Get contour center for mouse control
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        contour_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Smooth gesture detection
                    self.gesture_history.append(current_gesture)
                    if len(self.gesture_history) >= CONFIG['smoothing_window']:
                        most_common = max(set(self.gesture_history), key=list(self.gesture_history).count)
                        
                        # Execute action if gesture is stable
                        if most_common == current_gesture and time.time() - self.last_gesture_time > CONFIG['gesture_hold_time']:
                            self.execute_gesture_action(current_gesture, contour_center)
                            self.update_gesture_stats(current_gesture)
                            self.last_gesture_time = time.time()
                            self.current_gesture = current_gesture
                else:
                    self.prev_mouse_pos = (0, 0)  # Reset mouse position
                
                # Draw UI
                frame = self.draw_enhanced_ui(frame, current_gesture, hand_contours)
                
                # Show both original and processed frames
                cv2.imshow('Gesture Text Writer', frame)
                cv2.imshow('Background Mask', fg_mask)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.save_session_data()
            print(f"Session saved! Total gestures detected: {sum(self.gesture_stats.values())}")

if __name__ == "__main__":
    controller = GestureTextWriter()
    controller.run()
