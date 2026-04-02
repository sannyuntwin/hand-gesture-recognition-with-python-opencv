import math
import random
import time
from pathlib import Path
import ctypes
import subprocess
import winsound

import cv2
import mediapipe as mp
import numpy as np


WINDOW_NAME = "Ryomen Sukuna Power"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
ACTIVATION_SECONDS = 0.4
POWER_FADE_SECONDS = 1.2
DOMAIN_FLASH_SECONDS = 0.45
TIP_IDS = [4, 8, 12, 16, 20]
INDEX_TIP_ID = 8
INDEX_PIP_ID = 6
WRIST_ID = 0
FINGER_FOLDED_MARGIN = 0.035
SOUND_EFFECT_PATH = Path("sound-effect") / "dragon-studio-power-off-386180.mp3"


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


def hand_center_px(hand_landmarks, width, height):
    points = np.array([(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark], dtype=np.float32)
    center = np.mean(points, axis=0)
    return int(center[0]), int(center[1])


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
    left_fingers = count_fingers(hand_a, "Left")
    right_fingers = count_fingers(hand_b, "Right")
    if not (is_single_index_pose(left_fingers) and is_single_index_pose(right_fingers)):
        if not (index_is_dominant(hand_a) and index_is_dominant(hand_b)):
            return False, None, None

    left_index_tip = landmark_px(hand_a, INDEX_TIP_ID, width, height)
    right_index_tip = landmark_px(hand_b, INDEX_TIP_ID, width, height)
    left_index_pip = landmark_px(hand_a, INDEX_PIP_ID, width, height)
    right_index_pip = landmark_px(hand_b, INDEX_PIP_ID, width, height)
    left_wrist = landmark_px(hand_a, WRIST_ID, width, height)
    right_wrist = landmark_px(hand_b, WRIST_ID, width, height)

    fingertip_gap = np.linalg.norm(left_index_tip - right_index_tip)
    wrist_gap = np.linalg.norm(left_wrist - right_wrist)
    tip_height_gap = abs(left_index_tip[1] - right_index_tip[1])
    left_index_vertical = left_index_pip[1] - left_index_tip[1]
    right_index_vertical = right_index_pip[1] - right_index_tip[1]

    pose_ok = (
        fingertip_gap < max(120, width * 0.11)
        and wrist_gap < width * 0.62
        and tip_height_gap < max(85, height * 0.12)
        and left_index_vertical > height * 0.02
        and right_index_vertical > height * 0.02
    )

    center = ((left_index_tip + right_index_tip) / 2.0).astype(np.int32)
    debug = {
        "fingertip_gap": float(fingertip_gap),
        "tip_gap_ok": fingertip_gap < max(120, width * 0.11),
        "wrist_gap_ok": wrist_gap < width * 0.62,
        "height_ok": tip_height_gap < max(85, height * 0.12),
        "left_index_ok": left_index_vertical > height * 0.02,
        "right_index_ok": right_index_vertical > height * 0.02,
        "left_fingers": left_fingers,
        "right_fingers": right_fingers,
        "left_tip": (int(left_index_tip[0]), int(left_index_tip[1])),
        "right_tip": (int(right_index_tip[0]), int(right_index_tip[1])),
    }
    return pose_ok, (int(center[0]), int(center[1])), debug


def draw_pose_guide(frame, left_tip, right_tip, pose_center, hold_progress, pose_detected):
    guide_color = (90, 220, 255) if pose_detected else (90, 150, 220)
    cv2.circle(frame, left_tip, 12, guide_color, 2, cv2.LINE_AA)
    cv2.circle(frame, right_tip, 12, guide_color, 2, cv2.LINE_AA)
    cv2.line(frame, left_tip, right_tip, guide_color, 2, cv2.LINE_AA)
    cv2.circle(frame, pose_center, 26, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.ellipse(frame, pose_center, (32, 32), 0, -90, int(-90 + 360 * hold_progress), (0, 255, 255), 4)


def draw_detection_hints(frame, debug_info):
    if debug_info is None:
        return

    x = 30
    y = 170
    color_good = (120, 255, 180)
    color_bad = (130, 190, 255)

    hints = [
        ("Left hand: index only", debug_info["left_fingers"][1] == 1 and debug_info["left_fingers"][2] == 0 and debug_info["left_fingers"][3] == 0 and debug_info["left_fingers"][4] == 0),
        ("Right hand: index only", debug_info["right_fingers"][1] == 1 and debug_info["right_fingers"][2] == 0 and debug_info["right_fingers"][3] == 0 and debug_info["right_fingers"][4] == 0),
        ("Bring fingertips closer", debug_info["tip_gap_ok"]),
        ("Keep fingertips level", debug_info["height_ok"]),
    ]

    for label, ok in hints:
        draw_text(frame, label, (x, y), color_good if ok else color_bad, scale=0.6, thickness=2)
        y += 24


def draw_debug_metrics(frame, debug_info):
    if debug_info is None:
        return

    panel = frame.copy()
    x = 24
    y = 260
    w = 360
    h = 170
    cv2.rectangle(panel, (x, y), (x + w, y + h), (8, 12, 20), -1)
    frame[:] = cv2.addWeighted(panel, 0.72, frame, 0.28, 0)

    draw_text(frame, "Debug", (x + 14, y + 28), (255, 255, 255), scale=0.7, thickness=2)

    metrics = [
        f"Left fingers: {debug_info['left_fingers']}",
        f"Right fingers: {debug_info['right_fingers']}",
        f"Tip gap: {debug_info['fingertip_gap']:.1f}",
        f"Tip gap ok: {debug_info['tip_gap_ok']}",
        f"Wrist gap ok: {debug_info['wrist_gap_ok']}",
        f"Level ok: {debug_info['height_ok']}",
        f"Left index up: {debug_info['left_index_ok']}",
        f"Right index up: {debug_info['right_index_ok']}",
    ]

    line_y = y + 56
    for line in metrics:
        is_ok = line.endswith("True")
        is_bad = line.endswith("False")
        color = (220, 220, 220)
        if is_ok:
            color = (120, 255, 180)
        elif is_bad:
            color = (130, 190, 255)
        draw_text(frame, line, (x + 14, line_y), color, scale=0.54, thickness=1)
        line_y += 18


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
    alias = "sukuna_power"
    try:
        ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, None)
        ctypes.windll.winmm.mciSendStringW(
            f'open "{sound_path.resolve()}" type mpegvideo alias {alias}',
            None,
            0,
            None,
        )
        ctypes.windll.winmm.mciSendStringW(f'play {alias} from 0', None, 0, None)
        return
    except Exception:
        pass

    if sound_path.suffix.lower() == ".wav":
        try:
            winsound.PlaySound(str(sound_path.resolve()), winsound.SND_ASYNC | winsound.SND_FILENAME)
            return
        except Exception:
            pass

    # Fallback for MP3 playback on Windows when MCI is unreliable.
    escaped_path = str(sound_path.resolve()).replace("'", "''")
    powershell_script = (
        "Add-Type -AssemblyName presentationCore; "
        "$player = New-Object System.Windows.Media.MediaPlayer; "
        f"$player.Open([Uri]'{escaped_path}'); "
        "$player.Volume = 0.9; "
        "$player.Play(); "
        "Start-Sleep -Seconds 4"
    )
    try:
        subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-WindowStyle",
                "Hidden",
                "-Command",
                powershell_script,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def draw_malevolent_shrine_background(frame, intensity, rng):
    if intensity <= 0.0:
        return

    h, w = frame.shape[:2]
    scene = np.zeros_like(frame)

    for row in range(h):
        t = row / max(1, h - 1)
        red = int(35 + 80 * t)
        green = int(12 + 18 * t)
        blue = int(18 + 35 * t)
        scene[row, :, :] = (blue, green, red)

    horizon = int(h * 0.72)
    cv2.rectangle(scene, (0, horizon), (w, h), (18, 18, 26), -1)

    shrine_width = int(w * 0.24)
    shrine_height = int(h * 0.42)
    shrine_x = w // 2 - shrine_width // 2
    shrine_y = horizon - shrine_height + 30
    cv2.rectangle(scene, (shrine_x, shrine_y), (shrine_x + shrine_width, horizon), (32, 24, 32), -1)
    cv2.rectangle(scene, (shrine_x + 22, shrine_y + 30), (shrine_x + shrine_width - 22, horizon - 22), (15, 15, 18), -1)
    roof = np.array([
        (shrine_x - 36, shrine_y + 22),
        (w // 2, shrine_y - 42),
        (shrine_x + shrine_width + 36, shrine_y + 22),
    ], dtype=np.int32)
    cv2.fillConvexPoly(scene, roof, (46, 28, 36))
    cv2.line(scene, (shrine_x - 30, shrine_y + 28), (shrine_x + shrine_width + 30, shrine_y + 28), (70, 50, 58), 3, cv2.LINE_AA)

    for idx in range(7):
        col_x = int((idx + 0.5) * w / 7)
        top = horizon - rng.randint(20, 120)
        cv2.line(scene, (col_x, top), (col_x, h), (28, 24, 30), 2, cv2.LINE_AA)

    for _ in range(18):
        ember_x = rng.randint(0, w - 1)
        ember_y = rng.randint(0, horizon)
        ember_r = rng.randint(1, 3)
        ember_color = (40, rng.randint(60, 110), rng.randint(180, 255))
        cv2.circle(scene, (ember_x, ember_y), ember_r, ember_color, -1, cv2.LINE_AA)

    frame[:] = cv2.addWeighted(scene, 0.22 + 0.33 * intensity, frame, 0.78 - 0.33 * intensity, 0)


def draw_power_effect(frame, center, power_strength, rng):
    if power_strength <= 0.0:
        return

    h, w = frame.shape[:2]
    cx, cy = center
    overlay = frame.copy()

    aura_color = (40, 40, 255)
    ring_color = (0, 220, 255)
    slash_color = (255, 255, 255)

    radius = int(90 + 90 * power_strength)
    cv2.circle(overlay, (cx, cy), radius, aura_color, -1)
    frame[:] = cv2.addWeighted(overlay, 0.12 + power_strength * 0.18, frame, 0.88 - power_strength * 0.18, 0)

    for idx in range(3):
        current_radius = radius + idx * 24
        thickness = max(2, int(4 * power_strength))
        cv2.circle(frame, (cx, cy), current_radius, ring_color, thickness, cv2.LINE_AA)

    for _ in range(8):
        angle = rng.uniform(0, math.tau)
        length = rng.randint(80, 190)
        x2 = int(cx + math.cos(angle) * length)
        y2 = int(cy + math.sin(angle) * length)
        cv2.line(frame, (cx, cy), (x2, y2), slash_color, 2, cv2.LINE_AA)

    for _ in range(4):
        start_x = int(cx + rng.uniform(-120, 120))
        start_y = int(cy + rng.uniform(-120, 120))
        end_x = start_x + rng.randint(-70, 70)
        end_y = start_y + rng.randint(80, 180)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (30, 110, 255), 4, cv2.LINE_AA)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (120, 220, 255), 2, cv2.LINE_AA)

    glow = np.zeros_like(frame)
    top_left = (max(0, cx - 110), max(0, cy - 110))
    bottom_right = (min(w - 1, cx + 110), min(h - 1, cy + 110))
    cv2.rectangle(glow, top_left, bottom_right, (30, 30, 180), -1)
    frame[:] = cv2.addWeighted(frame, 1.0, glow, 0.16 * power_strength, 0)


def draw_domain_flash(frame, flash_strength):
    if flash_strength <= 0.0:
        return

    overlay = frame.copy()
    h, w = frame.shape[:2]
    tint = (30, 30, 160)
    cv2.rectangle(overlay, (0, 0), (w, h), tint, -1)
    for y in range(0, h, 36):
        cv2.line(overlay, (0, y), (w, y), (80, 80, 255), 1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.10 + 0.35 * flash_strength, frame, 0.90 - 0.35 * flash_strength, 0)


def draw_face_marks(frame, detection):
    if detection is None:
        return

    bbox = detection.location_data.relative_bounding_box
    h, w = frame.shape[:2]
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)
    if bw <= 0 or bh <= 0:
        return

    center_x = x + bw // 2
    center_y = y + bh // 2
    color = (20, 20, 20)
    accent = (0, 0, 0)

    forehead_y = y + int(bh * 0.18)
    eye_y = y + int(bh * 0.45)
    cheek_y = y + int(bh * 0.62)

    cv2.circle(frame, (center_x, forehead_y + 6), max(5, bw // 28), color, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (center_x - bw // 10, forehead_y + 6), (bw // 10, bh // 18), 0, 210, 340, accent, 2, cv2.LINE_AA)
    cv2.ellipse(frame, (center_x + bw // 10, forehead_y + 6), (bw // 10, bh // 18), 0, 200, 330, accent, 2, cv2.LINE_AA)

    left_cheek = np.array([
        (center_x - bw // 4, cheek_y),
        (center_x - bw // 3, cheek_y + bh // 18),
        (center_x - bw // 4, cheek_y + bh // 10),
    ], dtype=np.int32)
    right_cheek = np.array([
        (center_x + bw // 4, cheek_y),
        (center_x + bw // 3, cheek_y + bh // 18),
        (center_x + bw // 4, cheek_y + bh // 10),
    ], dtype=np.int32)
    cv2.polylines(frame, [left_cheek], False, color, 3, cv2.LINE_AA)
    cv2.polylines(frame, [right_cheek], False, color, 3, cv2.LINE_AA)
    cv2.line(frame, (center_x - bw // 3, eye_y), (center_x - bw // 4, eye_y - bh // 16), color, 2, cv2.LINE_AA)
    cv2.line(frame, (center_x + bw // 3, eye_y), (center_x + bw // 4, eye_y - bh // 16), color, 2, cv2.LINE_AA)

    left_eye = (center_x - bw // 6, eye_y)
    right_eye = (center_x + bw // 6, eye_y)
    eye_overlay = frame.copy()
    glow_radius_x = max(10, bw // 12)
    glow_radius_y = max(6, bh // 18)
    cv2.ellipse(eye_overlay, left_eye, (glow_radius_x, glow_radius_y), 0, 0, 360, (40, 40, 255), -1, cv2.LINE_AA)
    cv2.ellipse(eye_overlay, right_eye, (glow_radius_x, glow_radius_y), 0, 0, 360, (40, 40, 255), -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(eye_overlay, 0.32, frame, 0.68, 0)
    cv2.ellipse(frame, left_eye, (max(5, bw // 20), max(2, bh // 30)), 0, 0, 360, (110, 180, 255), -1, cv2.LINE_AA)
    cv2.ellipse(frame, right_eye, (max(5, bw // 20), max(2, bh // 30)), 0, 0, 360, (110, 180, 255), -1, cv2.LINE_AA)


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    pose_start_time = None
    power_until = 0.0
    domain_flash_until = 0.0
    sound_played_for_activation = False
    last_center = (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)
    rng = random.Random()

    print("Ryomen Sukuna Power Demo")
    print("Use both hands")
    print("Make the Sukuna sign: both index fingers up and touching together")
    print("Keep the other fingers folded and hold briefly to activate power")
    print("Press 'q' to quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        face_results = face_detection.process(rgb_frame)

        h, w = frame.shape[:2]
        current_time = time.time()
        status_text = "Make the two-hand Sukuna pose"
        status_color = (200, 200, 200)
        debug_info = None

        if results.multi_hand_landmarks and results.multi_handedness:
            detected_hands = []
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness_label = hand_handedness.classification[0].label
                detected_hands.append((hand_landmarks, handedness_label))
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(detected_hands) >= 2:
                pose_detected, pose_center, debug_info = is_two_hand_sukuna_pose(
                    detected_hands[0][0],
                    detected_hands[1][0],
                    w,
                    h,
                )
                if pose_center is not None:
                    last_center = pose_center

                if pose_detected:
                    status_text = "Sukuna pose detected"
                    status_color = (90, 220, 255)
                    if pose_start_time is None:
                        pose_start_time = current_time
                    hold_progress = min(1.0, (current_time - pose_start_time) / ACTIVATION_SECONDS)
                    if current_time - pose_start_time >= ACTIVATION_SECONDS:
                        was_active = current_time < power_until
                        power_until = current_time + POWER_FADE_SECONDS
                        domain_flash_until = current_time + DOMAIN_FLASH_SECONDS
                        status_text = "Power unleashed"
                        status_color = (60, 60, 255)
                        if not was_active:
                            sound_played_for_activation = False
                else:
                    pose_start_time = None
                    hold_progress = 0.0
                    status_text = "Match the guide and bring both index fingers together"
                    status_color = (120, 180, 255)

                if debug_info is not None:
                    draw_pose_guide(
                        frame,
                        debug_info["left_tip"],
                        debug_info["right_tip"],
                        last_center,
                        hold_progress,
                        pose_detected,
                    )
            else:
                pose_start_time = None
                status_text = "Show two hands clearly in frame"
                status_color = (120, 180, 255)
        else:
            pose_start_time = None

        flash_strength = max(0.0, min(1.0, (domain_flash_until - current_time) / DOMAIN_FLASH_SECONDS))
        if flash_strength > 0.0:
            draw_domain_flash(frame, flash_strength)

        power_strength = max(0.0, min(1.0, (power_until - current_time) / POWER_FADE_SECONDS))
        if power_strength > 0.0:
            draw_malevolent_shrine_background(frame, power_strength, rng)

        if face_results.detections:
            draw_face_marks(frame, face_results.detections[0])

        if power_strength > 0.0:
            if not sound_played_for_activation:
                play_sound_effect(SOUND_EFFECT_PATH)
                sound_played_for_activation = True
            draw_power_effect(frame, last_center, power_strength, rng)
            draw_text(frame, "CURSED POWER", (30, 110), (40, 40, 255), scale=1.2, thickness=3)
            draw_text(frame, "DOMAIN EXPANSION", (30, 150), (120, 220, 255), scale=0.9, thickness=2)
        else:
            sound_played_for_activation = False
            draw_detection_hints(frame, debug_info)
            draw_debug_metrics(frame, debug_info)

        top_overlay = frame.copy()
        cv2.rectangle(top_overlay, (16, 16), (430, 140), (10, 10, 18), -1)
        frame[:] = cv2.addWeighted(top_overlay, 0.72, frame, 0.28, 0)

        draw_text(frame, "Ryomen Sukuna", (30, 52), (255, 255, 255), scale=1.0, thickness=2)
        draw_text(frame, "Pose: both index fingers up and almost touching", (30, 82), (180, 180, 180), scale=0.7, thickness=1)
        draw_text(frame, status_text, (30, 126), status_color, scale=0.9, thickness=2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    face_detection.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
