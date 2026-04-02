import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


WINDOW_NAME = "Hand Swipe Slides"
SLIDES_DIR = Path("slides")
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TIP_IDS = [4, 8, 12, 16, 20]
FINGER_FOLDED_MARGIN = 0.03
OPEN_HAND_REQUIRED_FINGERS = 4
SWIPE_DISTANCE_PX = 150
SWIPE_TIME_SECONDS = 0.7
SWIPE_COOLDOWN_SECONDS = 0.9
TRAIL_LENGTH = 12
MIN_SWIPE_VELOCITY = 180.0


def draw_text(frame, text, origin, color, scale=1.0, thickness=2):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def count_fingers(hand_landmarks, handedness_label):
    fingers = []

    thumb_tip_x = hand_landmarks.landmark[TIP_IDS[0]].x
    thumb_ip_x = hand_landmarks.landmark[TIP_IDS[0] - 1].x
    if handedness_label == "Right":
        fingers.append(1 if thumb_tip_x < thumb_ip_x else 0)
    else:
        fingers.append(1 if thumb_tip_x > thumb_ip_x else 0)

    for i in range(1, 5):
        tip_y = hand_landmarks.landmark[TIP_IDS[i]].y
        pip_y = hand_landmarks.landmark[TIP_IDS[i] - 2].y
        fingers.append(1 if tip_y < pip_y - FINGER_FOLDED_MARGIN else 0)

    return fingers


def is_open_hand(fingers):
    return sum(fingers) >= OPEN_HAND_REQUIRED_FINGERS


def landmark_px(hand_landmarks, landmark_id, width, height):
    landmark = hand_landmarks.landmark[landmark_id]
    return int(landmark.x * width), int(landmark.y * height)


def load_slides():
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    slides = []
    if not SLIDES_DIR.exists():
        return slides

    for path in sorted(SLIDES_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            image = cv2.imread(str(path))
            if image is not None:
                title = path.stem.replace("_", " ").replace("-", " ")
                slides.append((title, image))
    return slides


def resize_image_to_fit(image, max_width, max_height):
    src_h, src_w = image.shape[:2]
    scale = min(max_width / src_w, max_height / src_h)
    new_width = max(1, int(src_w * scale))
    new_height = max(1, int(src_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def draw_slide(frame, slide_title, slide_image, slide_index, slide_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (8, 12, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.86, frame, 0.14, 0)

    panel_x = 42
    panel_y = 42
    panel_w = w - 84
    panel_h = h - 84
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (24, 28, 36), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (88, 96, 116), 2)

    draw_text(frame, "Hand Swipe Presentation", (panel_x + 28, panel_y + 42), (245, 247, 250), scale=1.0, thickness=2)
    draw_text(frame, slide_title, (panel_x + 28, panel_y + 82), (255, 190, 120), scale=0.9, thickness=2)
    draw_text(frame, f"Slide {slide_index + 1}/{slide_count}", (panel_x + 28, panel_y + panel_h - 24), (210, 214, 224), scale=0.7, thickness=2)

    image_x = panel_x + 28
    image_y = panel_y + 112
    image_w = panel_w - 56
    image_h = panel_h - 156
    resized = resize_image_to_fit(slide_image, image_w, image_h)
    rh, rw = resized.shape[:2]
    offset_x = image_x + (image_w - rw) // 2
    offset_y = image_y + (image_h - rh) // 2
    cv2.rectangle(frame, (image_x, image_y), (image_x + image_w, image_y + image_h), (12, 15, 22), -1)
    frame[offset_y:offset_y + rh, offset_x:offset_x + rw] = resized


def draw_camera_inset(frame, camera_frame):
    inset_w = 320
    inset_h = 180
    inset_x = frame.shape[1] - inset_w - 26
    inset_y = 26
    resized = cv2.resize(camera_frame, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    frame[inset_y:inset_y + inset_h, inset_x:inset_x + inset_w] = resized
    cv2.rectangle(frame, (inset_x, inset_y), (inset_x + inset_w, inset_y + inset_h), (255, 255, 255), 2)
    draw_text(frame, "Live Camera", (inset_x + 12, inset_y + 28), (255, 255, 255), scale=0.6, thickness=1)


def draw_swipe_trail(frame, points):
    if len(points) < 2:
        return

    for idx in range(1, len(points)):
        color_strength = idx / max(1, len(points) - 1)
        color = (int(80 + 120 * color_strength), int(120 + 100 * color_strength), 255)
        cv2.line(frame, points[idx - 1], points[idx], color, 2, cv2.LINE_AA)
    cv2.circle(frame, points[-1], 10, (0, 255, 255), -1, cv2.LINE_AA)


def main():
    slides = load_slides()
    if not slides:
        print(f"No slide images found in {SLIDES_DIR.resolve()}")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    slide_index = 0
    swipe_points = []
    swipe_start_time = None
    last_swipe_time = 0.0
    status_text = "Show an open hand and swipe left or right"
    fullscreen_enabled = False

    print("Hand Swipe Slides")
    print(f"Loaded slides: {len(slides)} from {SLIDES_DIR.resolve()}")
    print("Open your hand with 5 fingers, then swipe right for next slide")
    print("Open your hand with 5 fingers, then swipe left for previous slide")
    print("Press A for previous slide, D for next slide, F for fullscreen, Q to quit")

    while cap.isOpened():
        success, camera_frame = cap.read()
        if not success:
            continue

        camera_frame = cv2.flip(camera_frame, 1)
        rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        h, w = camera_frame.shape[:2]
        current_time = time.time()

        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        draw_slide(frame, slides[slide_index][0], slides[slide_index][1], slide_index, len(slides))

        hand_is_open = False
        cursor_point = None

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = results.multi_handedness[0].classification[0].label
            fingers = count_fingers(hand_landmarks, handedness_label)
            hand_is_open = is_open_hand(fingers)
            cursor_point = landmark_px(hand_landmarks, 9, w, h)

            mp_draw.draw_landmarks(camera_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if hand_is_open:
                status_text = "Open hand detected - swipe to change slide"
                if swipe_start_time is None:
                    swipe_start_time = current_time
                    swipe_points = [cursor_point]
                else:
                    swipe_points.append(cursor_point)
                    swipe_points = swipe_points[-TRAIL_LENGTH:]

                elapsed = current_time - swipe_start_time
                delta_x = swipe_points[-1][0] - swipe_points[0][0] if len(swipe_points) > 1 else 0
                swipe_velocity = delta_x / max(elapsed, 1e-6)
                can_trigger = current_time - last_swipe_time >= SWIPE_COOLDOWN_SECONDS
                if elapsed <= SWIPE_TIME_SECONDS and can_trigger:
                    if delta_x >= SWIPE_DISTANCE_PX or swipe_velocity >= MIN_SWIPE_VELOCITY:
                        slide_index = (slide_index + 1) % len(slides)
                        last_swipe_time = current_time
                        swipe_start_time = None
                        swipe_points = []
                        status_text = "Next slide"
                    elif delta_x <= -SWIPE_DISTANCE_PX or swipe_velocity <= -MIN_SWIPE_VELOCITY:
                        slide_index = (slide_index - 1) % len(slides)
                        last_swipe_time = current_time
                        swipe_start_time = None
                        swipe_points = []
                        status_text = "Previous slide"

                if elapsed > SWIPE_TIME_SECONDS:
                    swipe_start_time = current_time
                    swipe_points = [cursor_point]
            else:
                status_text = "Open your hand fully, then swipe"
                swipe_start_time = None
                swipe_points = []
        else:
            status_text = "Show one hand clearly to the camera"
            swipe_start_time = None
            swipe_points = []

        if cursor_point is not None and hand_is_open:
            draw_swipe_trail(camera_frame, swipe_points)

        draw_camera_inset(frame, camera_frame)
        draw_text(frame, status_text, (72, CAMERA_HEIGHT - 68), (120, 255, 180), scale=0.75, thickness=2)
        draw_text(frame, "Gesture: 5 fingers open + swipe left/right    F: fullscreen", (72, CAMERA_HEIGHT - 32), (220, 220, 220), scale=0.65, thickness=2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            slide_index = (slide_index - 1) % len(slides)
        if key == ord("d"):
            slide_index = (slide_index + 1) % len(slides)
        if key == ord("f"):
            fullscreen_enabled = not fullscreen_enabled
            mode = cv2.WINDOW_FULLSCREEN if fullscreen_enabled else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
