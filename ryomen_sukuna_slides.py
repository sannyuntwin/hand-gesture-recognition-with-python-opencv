import ctypes
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


WINDOW_NAME = "Ryomen Sukuna Slides"
SLIDES_DIR = Path("slides")
SOUND_EFFECT_PATH = Path("sound-effect") / "dragon-studio-power-off-386180.mp3"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
ACTIVATION_SECONDS = 0.35
ACTIVATION_COOLDOWN_SECONDS = 1.4
TIP_IDS = [4, 8, 12, 16, 20]
INDEX_TIP_ID = 8
INDEX_PIP_ID = 6
WRIST_ID = 0
FINGER_FOLDED_MARGIN = 0.035


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


def play_sound_effect(sound_path):
    if not sound_path.exists():
        return
    alias = "sukuna_slides"
    try:
        ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)
        ctypes.windll.winmm.mciSendStringW(
            f'open "{sound_path.resolve()}" type mpegvideo alias {alias}',
            None,
            0,
            None,
        )
        ctypes.windll.winmm.mciSendStringW(f"play {alias} from 0", None, 0, None)
    except Exception:
        pass


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


def is_single_index_pose(fingers):
    return fingers[1] == 1 and fingers[3] == 0 and fingers[4] == 0


def landmark_px(hand_landmarks, landmark_id, width, height):
    landmark = hand_landmarks.landmark[landmark_id]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def index_is_dominant(hand_landmarks):
    index_tip_y = hand_landmarks.landmark[INDEX_TIP_ID].y
    middle_tip_y = hand_landmarks.landmark[12].y
    ring_tip_y = hand_landmarks.landmark[16].y
    pinky_tip_y = hand_landmarks.landmark[20].y
    return (
        index_tip_y < ring_tip_y - 0.015
        and index_tip_y < pinky_tip_y - 0.015
        and index_tip_y < middle_tip_y + 0.035
    )


def is_two_hand_sukuna_pose(hand_a, hand_b, width, height):
    fingers_a = count_fingers(hand_a, "Left")
    fingers_b = count_fingers(hand_b, "Right")
    if not (is_single_index_pose(fingers_a) and is_single_index_pose(fingers_b)):
        if not (index_is_dominant(hand_a) and index_is_dominant(hand_b)):
            return False, None

    index_tip_a = landmark_px(hand_a, INDEX_TIP_ID, width, height)
    index_tip_b = landmark_px(hand_b, INDEX_TIP_ID, width, height)
    index_pip_a = landmark_px(hand_a, INDEX_PIP_ID, width, height)
    index_pip_b = landmark_px(hand_b, INDEX_PIP_ID, width, height)
    wrist_a = landmark_px(hand_a, WRIST_ID, width, height)
    wrist_b = landmark_px(hand_b, WRIST_ID, width, height)

    fingertip_gap = np.linalg.norm(index_tip_a - index_tip_b)
    wrist_gap = np.linalg.norm(wrist_a - wrist_b)
    tip_height_gap = abs(index_tip_a[1] - index_tip_b[1])
    vertical_a = index_pip_a[1] - index_tip_a[1]
    vertical_b = index_pip_b[1] - index_tip_b[1]

    pose_ok = (
        fingertip_gap < max(120, width * 0.11)
        and wrist_gap < width * 0.62
        and tip_height_gap < max(85, height * 0.12)
        and vertical_a > height * 0.02
        and vertical_b > height * 0.02
    )
    center = ((index_tip_a + index_tip_b) / 2.0).astype(np.int32)
    return pose_ok, (int(center[0]), int(center[1]))


def resize_image_to_fit(image, max_width, max_height):
    src_h, src_w = image.shape[:2]
    scale = min(max_width / src_w, max_height / src_h)
    new_width = max(1, int(src_w * scale))
    new_height = max(1, int(src_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def load_slides():
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not SLIDES_DIR.exists():
        return []

    slides = []
    for path in sorted(SLIDES_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            image = cv2.imread(str(path))
            if image is not None:
                slides.append((path.stem.replace("_", " ").replace("-", " "), image))
    return slides


def draw_slide_canvas(frame, slide_title, slide_image):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 12, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    panel_margin = 40
    panel_x = panel_margin
    panel_y = 40
    panel_w = w - panel_margin * 2
    panel_h = h - 80
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (24, 28, 38), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (70, 80, 100), 2)

    draw_text(frame, "Malevolent Shrine Presentation", (panel_x + 28, panel_y + 40), (245, 247, 250), scale=1.0, thickness=2)
    draw_text(frame, slide_title, (panel_x + 28, panel_y + 84), (255, 170, 110), scale=0.9, thickness=2)

    image_x = panel_x + 28
    image_y = panel_y + 110
    image_w = panel_w - 56
    image_h = panel_h - 145
    resized = resize_image_to_fit(slide_image, image_w, image_h)
    image_h2, image_w2 = resized.shape[:2]
    offset_x = image_x + (image_w - image_w2) // 2
    offset_y = image_y + (image_h - image_h2) // 2
    cv2.rectangle(frame, (image_x, image_y), (image_x + image_w, image_y + image_h), (12, 14, 18), -1)
    frame[offset_y:offset_y + image_h2, offset_x:offset_x + image_w2] = resized


def draw_camera_inset(frame, camera_frame):
    inset_w = 300
    inset_h = 170
    inset_x = frame.shape[1] - inset_w - 24
    inset_y = 24
    resized = cv2.resize(camera_frame, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    frame[inset_y:inset_y + inset_h, inset_x:inset_x + inset_w] = resized
    cv2.rectangle(frame, (inset_x, inset_y), (inset_x + inset_w, inset_y + inset_h), (255, 255, 255), 2)
    draw_text(frame, "Live Hand Cam", (inset_x + 12, inset_y + 28), (255, 255, 255), scale=0.6, thickness=1)


def draw_activation_effect(frame, center, progress):
    color = (40, 60, 255)
    radius = int(40 + 80 * progress)
    cv2.circle(frame, center, radius, color, 3, cv2.LINE_AA)
    cv2.circle(frame, center, radius + 18, (80, 220, 255), 2, cv2.LINE_AA)
    cv2.ellipse(frame, center, (radius + 28, radius + 28), 0, -90, int(-90 + 360 * progress), (0, 255, 255), 4)


def main():
    slides = load_slides()
    if not slides:
        print(f"No slide images found in {SLIDES_DIR.resolve()}")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    slide_index = 0
    pose_start_time = None
    last_activation_time = 0.0
    activation_center = (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)
    activation_flash_until = 0.0

    print("Ryomen Sukuna Slides")
    print(f"Loaded slides: {len(slides)} from {SLIDES_DIR.resolve()}")
    print("Use the two-hand Sukuna sign to go to the next slide")
    print("Press A for previous slide, D for next slide, Q to quit")

    while cap.isOpened():
        success, camera_frame = cap.read()
        if not success:
            continue

        camera_frame = cv2.flip(camera_frame, 1)
        rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        slide_title, slide_image = slides[slide_index]
        draw_slide_canvas(frame, slide_title, slide_image)

        current_time = time.time()
        status_text = "Make the Sukuna sign to advance"
        status_color = (220, 220, 220)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                mp_draw.draw_landmarks(camera_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            pose_detected, pose_center = is_two_hand_sukuna_pose(
                results.multi_hand_landmarks[0],
                results.multi_hand_landmarks[1],
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            )
            if pose_center is not None:
                activation_center = pose_center

            if pose_detected:
                status_text = "Sukuna sign detected"
                status_color = (120, 255, 180)
                if pose_start_time is None:
                    pose_start_time = current_time
                hold_progress = min(1.0, (current_time - pose_start_time) / ACTIVATION_SECONDS)
                draw_activation_effect(frame, activation_center, hold_progress)

                can_activate = current_time - last_activation_time >= ACTIVATION_COOLDOWN_SECONDS
                if hold_progress >= 1.0 and can_activate:
                    slide_index = (slide_index + 1) % len(slides)
                    last_activation_time = current_time
                    activation_flash_until = current_time + 0.55
                    play_sound_effect(SOUND_EFFECT_PATH)
                    status_text = f"Slide opened: {slides[slide_index][0]}"
                    status_color = (255, 170, 110)
            else:
                pose_start_time = None
                status_text = "Show both hands and bring the index fingers together"
                status_color = (130, 190, 255)
        else:
            pose_start_time = None
            status_text = "Show two hands clearly to the camera"
            status_color = (130, 190, 255)

        if current_time < activation_flash_until:
            flash_overlay = frame.copy()
            cv2.rectangle(flash_overlay, (0, 0), (frame.shape[1], frame.shape[0]), (40, 40, 160), -1)
            frame[:] = cv2.addWeighted(flash_overlay, 0.15, frame, 0.85, 0)

        draw_camera_inset(frame, camera_frame)
        draw_text(frame, f"Slide {slide_index + 1}/{len(slides)}", (72, CAMERA_HEIGHT - 34), (255, 255, 255), scale=0.75, thickness=2)
        draw_text(frame, status_text, (72, CAMERA_HEIGHT - 72), status_color, scale=0.75, thickness=2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            slide_index = (slide_index - 1) % len(slides)
        if key == ord("d"):
            slide_index = (slide_index + 1) % len(slides)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
