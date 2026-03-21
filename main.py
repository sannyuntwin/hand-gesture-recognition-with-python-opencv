# ==============================
# 🖐️ INTEGRATED GESTURE MENU SYSTEM
# ==============================
# All modules run in the same window - no separate processes
# Choose different modules by pointing and selecting
# ==============================

import cv2
import numpy as np
import time
import pyautogui
from collections import deque
import json
import math
import os

# Configuration
CONFIG = {
    'smoothing_window': 3,
    'gesture_hold_time': 0.5,
    'mouse_sensitivity': 1.5,
    'save_gestures': True,
    'camera_index': 0
}

class MenuModule:
    def __init__(self, name, description, icon):
        self.name = name
        self.description = description
        self.icon = icon
        self.x = 0
        self.y = 0
        self.width = 200
        self.height = 150
        self.selected = False

class IntegratedGestureMenu:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(CONFIG['camera_index'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Background subtractor for hand detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Tracking variables
        self.gesture_history = deque(maxlen=CONFIG['smoothing_window'])
        self.last_gesture_time = time.time()
        self.current_gesture = "No Hand Detected"
        self.mouse_position = (0, 0)
        self.prev_mouse_pos = (0, 0)
        
        # Menu variables
        self.current_module_index = 0
        self.hovered_module = None
        self.menu_active = True
        self.current_mode = "menu"  # "menu", "text_writing", "mouse_control", etc.
        
        # Gesture counters
        self.gesture_stats = {}
        self.session_start = time.time()
        
        # Hand detection parameters
        self.hand_contour_min_area = 3000
        self.hand_contour_max_area = 80000
        
        # Text writing variables
        self.keyboard = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['SPACE', 'BACKSPACE', 'ENTER']
        ]
        self.current_key_pos = [0, 0]
        self.current_key = self.keyboard[0][0]
        self.last_text_time = time.time()
        
        # Mouse control variables
        self.mouse_sensitivity = 2.0
        
        # Create menu modules
        self.create_menu_modules()
        
    def create_menu_modules(self):
        """Create available menu modules"""
        self.modules = [
            MenuModule("Gesture Detection", "Basic hand gesture recognition", "👋"),
            MenuModule("Text Writing", "Type text with hand gestures", "⌨️"),
            MenuModule("Mouse Control", "Control cursor with gestures", "🖱️"),
            MenuModule("Presentation", "Control slides hands-free", "📊"),
            MenuModule("Volume Control", "Adjust system volume", "🔊"),
            MenuModule("Sign Language", "Learn sign language basics", "🤟"),
            MenuModule("Games", "Play gesture-based games", "🎮"),
            MenuModule("Settings", "Configure the system", "⚙️"),
            MenuModule("Back to Menu", "Return to main menu", "🔙")
        ]
        
        # Arrange modules in a grid
        self.arrange_modules_grid()
    
    def arrange_modules_grid(self):
        """Arrange modules in a 3x3 grid"""
        cols = 3
        rows = 3
        start_x = 50
        start_y = 100
        spacing_x = 250
        spacing_y = 200
        
        for i, module in enumerate(self.modules):
            row = i // cols
            col = i % cols
            module.x = start_x + col * spacing_x
            module.y = start_y + row * spacing_y
    
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
            if angle < 90 and d > 8000:
                finger_count += 1
        
        # Special case for pointing gesture (index finger extended)
        if finger_count == 1:
            # Get contour center and topmost point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Find the topmost point of the contour
                contour_points = contour.reshape(-1, 2)
                topmost_idx = np.argmin(contour_points[:, 1])
                topmost_point = tuple(contour_points[topmost_idx])
                
                # Check if topmost point is significantly above center
                if topmost_point[1] < cy - 30:
                    return 1, contour
        
        # Adjust finger count
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
    
    def get_hovered_module(self, hand_center):
        """Get which module is being hovered over"""
        if not hand_center:
            return None
        
        x, y = hand_center
        for module in self.modules:
            if (module.x <= x <= module.x + module.width and
                module.y <= y <= module.y + module.height):
                return module
        return None
    
    def switch_mode(self, mode):
        """Switch between different modes"""
        self.current_mode = mode
        self.menu_active = (mode == "menu")
        print(f"🔄 Switched to {mode} mode")
    
    def execute_gesture_action(self, gesture, contour_center=None):
        """Execute actions based on detected gestures"""
        if self.current_mode == "menu":
            self.execute_menu_action(gesture, contour_center)
        elif self.current_mode == "text_writing":
            self.execute_text_writing_action(gesture, contour_center)
        elif self.current_mode == "mouse_control":
            self.execute_mouse_control_action(gesture, contour_center)
        elif self.current_mode == "presentation":
            self.execute_presentation_action(gesture, contour_center)
        elif self.current_mode == "volume_control":
            self.execute_volume_control_action(gesture, contour_center)
    
    def execute_menu_action(self, gesture, contour_center=None):
        """Execute menu actions"""
        if gesture == "Point" and contour_center:
            # Check which module is being hovered
            hovered = self.get_hovered_module(contour_center)
            if hovered:
                # Unselect all
                for module in self.modules:
                    module.selected = False
                # Select hovered
                hovered.selected = True
                self.current_module_index = self.modules.index(hovered)
                self.hovered_module = hovered
                
        elif gesture == "Fist":
            # Launch selected module
            if self.hovered_module:
                module_name = self.hovered_module.name
                print(f"🚀 Launching: {module_name}")
                
                if module_name == "Text Writing":
                    self.switch_mode("text_writing")
                elif module_name == "Mouse Control":
                    self.switch_mode("mouse_control")
                elif module_name == "Presentation":
                    self.switch_mode("presentation")
                elif module_name == "Volume Control":
                    self.switch_mode("volume_control")
                elif module_name == "Gesture Detection":
                    self.switch_mode("gesture_detection")
                elif module_name == "Sign Language":
                    self.switch_mode("sign_language")
                elif module_name == "Games":
                    self.switch_mode("games")
                elif module_name == "Settings":
                    self.switch_mode("settings")
                elif module_name == "Back to Menu":
                    self.switch_mode("menu")
    
    def execute_text_writing_action(self, gesture, contour_center=None):
        """Execute text writing actions"""
        if gesture == "Point" and contour_center:
            # Navigate keyboard based on hand position
            frame_center_x, frame_center_y = 320, 240
            cx, cy = contour_center
            
            if cx > frame_center_x + 50:
                self.navigate_keyboard("right")
            elif cx < frame_center_x - 50:
                self.navigate_keyboard("left")
            elif cy > frame_center_y + 50:
                self.navigate_keyboard("down")
            elif cy < frame_center_y - 50:
                self.navigate_keyboard("up")
        
        elif gesture == "Fist":
            # Type selected key
            if time.time() - self.last_text_time > 0.5:
                self.type_key(self.current_key)
                self.last_text_time = time.time()
        
        elif gesture == "Palm":
            # Go back to menu
            self.switch_mode("menu")
    
    def execute_mouse_control_action(self, gesture, contour_center=None):
        """Execute mouse control actions"""
        if gesture == "Point" and contour_center:
            # Move mouse based on hand position
            screen_width, screen_height = pyautogui.size()
            mouse_x = int((1 - contour_center[0] / 640) * screen_width)
            mouse_y = int((contour_center[1] / 480) * screen_height)
            pyautogui.moveTo(mouse_x, mouse_y)
        
        elif gesture == "Fist":
            # Click
            pyautogui.click()
        
        elif gesture == "Palm":
            # Go back to menu
            self.switch_mode("menu")
    
    def execute_presentation_action(self, gesture, contour_center=None):
        """Execute presentation control actions"""
        if gesture == "Peace":
            # Next slide
            pyautogui.press('right')
            print("➡️ Next slide")
        elif gesture == "Call_Me":
            # Previous slide
            pyautogui.press('left')
            print("⬅️ Previous slide")
        elif gesture == "Palm":
            # Go back to menu
            self.switch_mode("menu")
    
    def execute_volume_control_action(self, gesture, contour_center=None):
        """Execute volume control actions"""
        if gesture == "Three_Fingers":
            # Volume up
            pyautogui.press('volumeup')
            print("🔊 Volume up")
        elif gesture == "Two_Fingers":
            # Volume down
            pyautogui.press('volumedown')
            print("🔉 Volume down")
        elif gesture == "Palm":
            # Go back to menu
            self.switch_mode("menu")
    
    def navigate_keyboard(self, direction):
        """Navigate virtual keyboard"""
        rows = len(self.keyboard)
        cols = len(self.keyboard[0])
        
        if direction == "right":
            self.current_key_pos[1] = (self.current_key_pos[1] + 1) % cols
            if self.current_key_pos[1] >= len(self.keyboard[self.current_key_pos[0]]):
                self.current_key_pos[1] = 0
        elif direction == "left":
            self.current_key_pos[1] = (self.current_key_pos[1] - 1) % cols
            if self.current_key_pos[1] < 0:
                self.current_key_pos[1] = len(self.keyboard[self.current_key_pos[0]]) - 1
        elif direction == "down":
            self.current_key_pos[0] = (self.current_key_pos[0] + 1) % rows
            if self.current_key_pos[1] >= len(self.keyboard[self.current_key_pos[0]]):
                self.current_key_pos[1] = len(self.keyboard[self.current_key_pos[0]]) - 1
        elif direction == "up":
            self.current_key_pos[0] = (self.current_key_pos[0] - 1) % rows
            if self.current_key_pos[1] >= len(self.keyboard[self.current_key_pos[0]]):
                self.current_key_pos[1] = len(self.keyboard[self.current_key_pos[0]]) - 1
        
        self.current_key = self.keyboard[self.current_key_pos[0]][self.current_key_pos[1]]
    
    def type_key(self, key):
        """Type the selected key"""
        if key == 'SPACE':
            pyautogui.press('space')
            print("📝 Typed: SPACE")
        elif key == 'BACKSPACE':
            pyautogui.press('backspace')
            print("📝 Typed: BACKSPACE")
        elif key == 'ENTER':
            pyautogui.press('enter')
            print("📝 Typed: ENTER")
        else:
            pyautogui.typewrite(key)
            print(f"📝 Typed: {key}")
    
    def draw_ui(self, frame, gesture, contours=None):
        """Draw UI based on current mode"""
        if self.current_mode == "menu":
            return self.draw_menu_ui(frame, gesture, contours)
        elif self.current_mode == "text_writing":
            return self.draw_text_writing_ui(frame, gesture, contours)
        elif self.current_mode == "mouse_control":
            return self.draw_mouse_control_ui(frame, gesture, contours)
        elif self.current_mode == "presentation":
            return self.draw_presentation_ui(frame, gesture, contours)
        else:
            return self.draw_generic_ui(frame, gesture, contours)
    
    def draw_menu_ui(self, frame, gesture, contours=None):
        """Draw menu UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "GESTURE MENU SYSTEM", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Dynamic instruction
        if gesture == "Point":
            instruction = "✋ POINTING - Move hand to select module"
            instruction_color = (0, 255, 0)
        elif gesture == "Fist" and self.hovered_module:
            instruction = "✊ FIST - Launch selected module!"
            instruction_color = (0, 255, 255)
        else:
            instruction = "Point to select, Fist to launch"
            instruction_color = (200, 200, 200)
        
        cv2.putText(frame, instruction, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, instruction_color, 2)
        
        # Draw modules
        for module in self.modules:
            # Module background
            if module.selected:
                color = (0, 255, 0)
                thickness = 3
                bg_color = (0, 50, 0)
            elif module == self.hovered_module:
                color = (255, 255, 0)
                thickness = 3
                bg_color = (50, 50, 0)
            else:
                color = (100, 100, 100)
                thickness = 2
                bg_color = (30, 30, 30)
            
            cv2.rectangle(frame, (module.x, module.y), 
                        (module.x + module.width, module.y + module.height), 
                        bg_color, -1)
            cv2.rectangle(frame, (module.x, module.y), 
                        (module.x + module.width, module.y + module.height), 
                        color, thickness)
            
            # Module text
            cv2.putText(frame, module.icon, (module.x + 10, module.y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, module.name, (module.x + 10, module.y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, module.description[:20], (module.x + 10, module.y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw hand contours
        if contours:
            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if gesture == "Point":
                        cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
                        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
                        cv2.arrowedLine(frame, (cx, cy), (cx, cy - 30), (0, 255, 0), 3)
                        cv2.putText(frame, "POINTING", (cx - 40, cy - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)
        
        # Draw info
        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if self.hovered_module:
            cv2.putText(frame, f"Selected: {self.hovered_module.name}", (10, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "Make FIST to launch!", (10, h - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "👆 Point: Navigate | ✊ Fist: Launch | ✋ Palm: Back", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_text_writing_ui(self, frame, gesture, contours=None):
        """Draw text writing UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "TEXT WRITING MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Point: Navigate | Fist: Type | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw keyboard
        key_size = 40
        key_spacing = 45
        start_x = 50
        start_y = 100
        
        for row_idx, row in enumerate(self.keyboard):
            for col_idx, key in enumerate(row):
                x = start_x + col_idx * key_spacing
                y = start_y + row_idx * key_spacing
                
                # Highlight current key
                if row_idx == self.current_key_pos[0] and col_idx == self.current_key_pos[1]:
                    color = (0, 255, 0)
                    thickness = 3
                else:
                    color = (100, 100, 100)
                    thickness = 1
                
                cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), color, thickness)
                cv2.putText(frame, key[:3], (x + 5, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw current key
        cv2.putText(frame, f"Current: {self.current_key}", (10, h - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw gesture info
        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw hand contours
        if contours:
            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if gesture == "Point":
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        cv2.putText(frame, "POINT", (cx - 25, cy - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
        
        cv2.putText(frame, "Open any text editor to type!", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def draw_mouse_control_ui(self, frame, gesture, contours=None):
        """Draw mouse control UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "MOUSE CONTROL MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Point: Move cursor | Fist: Click | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw hand contours
        if contours:
            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if gesture == "Point":
                        cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
                        cv2.putText(frame, "MOVING CURSOR", (cx - 50, cy - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif gesture == "Fist":
                        cv2.circle(frame, (cx, cy), 12, (255, 0, 0), -1)
                        cv2.putText(frame, "CLICK!", (cx - 20, cy - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def draw_presentation_ui(self, frame, gesture, contours=None):
        """Draw presentation control UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "PRESENTATION CONTROL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Peace: Next slide | Call Me: Previous | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw large navigation buttons
        button_width = 150
        button_height = 100
        button_y = 200
        
        # Previous button
        prev_x = 100
        cv2.rectangle(frame, (prev_x, button_y), (prev_x + button_width, button_y + button_height), (100, 100, 255), -1)
        cv2.putText(frame, "PREVIOUS", (prev_x + 20, button_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Next button
        next_x = 400
        cv2.rectangle(frame, (next_x, button_y), (next_x + button_width, button_y + button_height), (100, 255, 100), -1)
        cv2.putText(frame, "NEXT", (next_x + 40, button_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw gesture info
        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def draw_generic_ui(self, frame, gesture, contours=None):
        """Draw generic UI for other modes"""
        h, w, _ = frame.shape
        
        if self.current_mode == "volume_control":
            return self.draw_volume_control_ui(frame, gesture, contours)
        elif self.current_mode == "gesture_detection":
            return self.draw_gesture_detection_ui(frame, gesture, contours)
        elif self.current_mode == "sign_language":
            return self.draw_sign_language_ui(frame, gesture, contours)
        elif self.current_mode == "games":
            return self.draw_games_ui(frame, gesture, contours)
        elif self.current_mode == "settings":
            return self.draw_settings_ui(frame, gesture, contours)
        else:
            return self.draw_default_ui(frame, gesture, contours)
    
    def draw_volume_control_ui(self, frame, gesture, contours=None):
        """Draw volume control UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "VOLUME CONTROL MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Three Fingers: Volume Up | Two Fingers: Volume Down | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw volume bar
        bar_width = 400
        bar_height = 40
        bar_x = (w - bar_width) // 2
        bar_y = h // 2 - 50
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Volume level (simulate current volume)
        volume_level = 70  # You can get real volume if needed
        fill_width = int((volume_level / 100) * bar_width)
        
        # Color based on volume level
        if volume_level < 30:
            color = (0, 0, 255)  # Red for low
        elif volume_level < 70:
            color = (0, 255, 255)  # Yellow for medium
        else:
            color = (0, 255, 0)  # Green for high
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Volume percentage
        cv2.putText(frame, f"Volume: {volume_level}%", (bar_x + bar_width//2 - 60, bar_y + bar_height//2 + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw speaker icon
        speaker_x = bar_x - 80
        speaker_y = bar_y + bar_height // 2
        
        # Speaker body
        cv2.rectangle(frame, (speaker_x, speaker_y - 20), (speaker_x + 30, speaker_y + 20), (255, 255, 255), -1)
        cv2.rectangle(frame, (speaker_x + 30, speaker_y - 30), (speaker_x + 40, speaker_y + 30), (255, 255, 255), -1)
        
        # Sound waves
        cv2.circle(frame, (speaker_x + 60, speaker_y), 8, (255, 255, 255), 2)
        cv2.circle(frame, (speaker_x + 80, speaker_y), 12, (255, 255, 255), 2)
        cv2.circle(frame, (speaker_x + 100, speaker_y), 16, (255, 255, 255), 2)
        
        # Draw gesture info
        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Volume change indicator
        if gesture == "Three_Fingers":
            cv2.putText(frame, "🔊 VOLUME UP", (w//2 - 80, h//2 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif gesture == "Two_Fingers":
            cv2.putText(frame, "🔉 VOLUME DOWN", (w//2 - 80, h//2 + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        
        return frame
    
    def draw_gesture_detection_ui(self, frame, gesture, contours=None):
        """Draw gesture detection UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "GESTURE DETECTION MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Practice your hand gestures | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw gesture guide
        gestures = [
            ("✊ Fist", 100, 120),
            ("✋ Palm", 250, 120),
            ("👆 Point", 400, 120),
            ("✌️ Peace", 100, 180),
            ("🤟 Three Fingers", 250, 180),
            ("🤙 Two Fingers", 400, 180)
        ]
        
        for gesture_name, x, y in gestures:
            cv2.putText(frame, gesture_name, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw current detected gesture
        cv2.putText(frame, f"Detected: {gesture}", (10, h - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw hand contours
        if contours:
            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)
        
        return frame
    
    def draw_sign_language_ui(self, frame, gesture, contours=None):
        """Draw sign language UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "SIGN LANGUAGE MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Learn sign language basics | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw sign language examples
        signs = [
            ("✌️ Peace = V Sign", 100, 120),
            ("👍 Thumbs Up = Good", 100, 160),
            ("🤙 OK Sign = Perfect", 100, 200),
            ("✋ Palm = Hello", 400, 120),
            ("✊ Fist = Stop", 400, 160),
            ("👆 Point = You", 400, 200)
        ]
        
        for sign_text, x, y in signs:
            cv2.putText(frame, sign_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw current detected gesture
        cv2.putText(frame, f"Your Gesture: {gesture}", (10, h - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame
    
    def draw_games_ui(self, frame, gesture, contours=None):
        """Draw games UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "GAMES MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Gesture-based games coming soon | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw coming soon message
        cv2.putText(frame, "🎮 COMING SOON", (w//2 - 120, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        
        # Draw game concepts
        games = [
            ("🎯 Target Practice", 100, 300),
            ("🎪 Catch the Moving Target", 100, 340),
            ("🏆 Gesture Challenge", 400, 300),
            ("🎮 Hand-Controlled Games", 400, 340)
        ]
        
        for game_text, x, y in games:
            cv2.putText(frame, game_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        return frame
    
    def draw_settings_ui(self, frame, gesture, contours=None):
        """Draw settings UI"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, "SETTINGS MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Configure system settings | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw settings options
        settings = [
            ("📹 Camera Settings", 100, 120),
            ("🎯 Sensitivity: Medium", 100, 160),
            ("⚡ Response Time: Fast", 100, 200),
            ("💾 Save Sessions: ON", 400, 120),
            ("🎨 UI Theme: Default", 400, 160),
            ("🔊 Sound Effects: ON", 400, 200)
        ]
        
        for setting_text, x, y in settings:
            cv2.putText(frame, setting_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def draw_default_ui(self, frame, gesture, contours=None):
        """Draw default UI for unknown modes"""
        h, w, _ = frame.shape
        
        # Draw title
        cv2.putText(frame, f"{self.current_mode.upper()} MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(frame, "Use gestures to control | Palm: Back to Menu", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw gesture info
        cv2.putText(frame, f"Gesture: {gesture}", (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def save_session_data(self):
        """Save session statistics to file"""
        if CONFIG['save_gestures']:
            session_data = {
                'duration': time.time() - self.session_start,
                'gesture_stats': self.gesture_stats,
                'timestamp': time.time(),
                'integrated_mode': True
            }
            
            with open('integrated_session.json', 'w') as f:
                json.dump(session_data, f, indent=2)
    
    def run(self):
        """Main execution loop"""
        print("🖐️ Integrated Gesture Menu System Started!")
        print("=" * 60)
        print("INSTRUCTIONS:")
        print("• All modules run in the SAME WINDOW!")
        print("• Point to select, Fist to launch")
        print("• Palm gesture to return to menu")
        print("• Press 'q' to quit")
        print("=" * 60)
        
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
                    
                    # Get contour center
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
                            # Update gesture statistics
                            if current_gesture not in self.gesture_stats:
                                self.gesture_stats[current_gesture] = 0
                            self.gesture_stats[current_gesture] += 1
                            self.last_gesture_time = time.time()
                            self.current_gesture = current_gesture
                else:
                    self.hovered_module = None
                
                # Draw UI
                frame = self.draw_ui(frame, current_gesture, hand_contours)
                
                # Show frame
                cv2.namedWindow('Integrated Gesture Menu', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Integrated Gesture Menu', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Integrated Gesture Menu', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.save_session_data()
            print(f"Session saved! Total gestures detected: {sum(self.gesture_stats.values())}")

if __name__ == "__main__":
    integrated_menu = IntegratedGestureMenu()
    integrated_menu.run()
