import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


WINDOW_NAME = "MediaPipe Drag and Drop Demo"
SLIDES_DIR = Path("slides")
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SMOOTHING = 0.18
SNAP_SPEED = 0.22
CARD_WIDTH = 92
CARD_HEIGHT = 56
CARD_GAP_X = 12
CARD_GAP_Y = 12
CARD_START_X = 20
CARD_START_Y = 198
CARD_COLUMNS = 2
CARD_ROWS = 3
PAGE_SIZE = CARD_COLUMNS * CARD_ROWS
GRAB_RADIUS = 65
HAND_LOST_GRACE_SECONDS = 0.35
HOVER_PICKUP_SECONDS = 0.35
HOVER_DROP_SECONDS = 0.12
DROP_FLASH_SECONDS = 0.55
SHOW_START_SCREEN = True
PREVIEW_X = 700
PREVIEW_Y = 95
PREVIEW_WIDTH = 500
PREVIEW_HEIGHT = 530
PREVIEW_SCALE = 0.72
ACCENT = (255, 168, 87)
PANEL_BG = (22, 26, 35)
PANEL_BG_ALT = (30, 36, 48)
PANEL_STROKE = (55, 64, 82)
TEXT_PRIMARY = (245, 247, 250)
TEXT_MUTED = (180, 188, 200)
SUCCESS = (129, 230, 160)
FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")
TEXT_CACHE = {}


class DraggableCard:
    def __init__(self, label, x, y, w, h, color, image=None):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.image = image
        self.placed_in = None
        self.target_position = None
        self.home_position = (x, y)

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def set_target(self, x, y):
        self.target_position = (x, y)

    def clear_target(self):
        self.target_position = None

    def reset_to_home(self):
        self.x, self.y = self.home_position
        self.clear_target()


class DropZone:
    def __init__(self, label, x, y, w, h, color):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def smooth_point(previous_point, target_point, alpha):
    if previous_point is None:
        return target_point

    x = int(previous_point[0] + (target_point[0] - previous_point[0]) * alpha)
    y = int(previous_point[1] + (target_point[1] - previous_point[1]) * alpha)
    return x, y


def load_font(size, bold=False):
    if ImageFont is None:
        return None
    font_path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if not font_path.exists():
        return None
    return ImageFont.truetype(str(font_path), size)


def get_cached_text_image(text, size, color, bold):
    cache_key = (text, size, color, bold)
    if cache_key in TEXT_CACHE:
        return TEXT_CACHE[cache_key]

    font = load_font(size, bold=bold)
    if font is None or Image is None or ImageDraw is None:
        return None

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=(color[2], color[1], color[0], 255))
    text_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    TEXT_CACHE[cache_key] = text_image
    return text_image


def draw_text(frame, text, origin, size, color, *, bold=False):
    text_image = get_cached_text_image(text, size, color, bold)
    if text_image is None:
        scale = max(0.5, size / 32.0)
        thickness = 2 if bold else 1
        font_face = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
        baseline_y = origin[1] + size
        cv2.putText(frame, text, (origin[0], baseline_y), font_face, scale, color, thickness, cv2.LINE_AA)
        return

    x, y = origin
    h, w = text_image.shape[:2]
    if x >= frame.shape[1] or y >= frame.shape[0]:
        return

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame.shape[1], x + w)
    y1 = min(frame.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return

    text_crop = text_image[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = text_crop[:, :, 3:4] / 255.0
    frame_region = frame[y0:y1, x0:x1].astype(np.float32)
    text_rgb = text_crop[:, :, :3].astype(np.float32)
    blended = text_rgb * alpha + frame_region * (1.0 - alpha)
    frame[y0:y1, x0:x1] = blended.astype(np.uint8)


def draw_panel(frame, x, y, w, h, fill_color, border_color=None, alpha=0.88, border_thickness=2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), fill_color, -1)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    if border_color is not None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thickness)


def draw_metric(frame, label, value, x, y):
    draw_text(frame, label, (x, y), 11, TEXT_MUTED, bold=False)
    draw_text(frame, value, (x, y + 14), 16, TEXT_PRIMARY, bold=True)


def resize_image_to_fit(image, max_width, max_height):
    if image is None or max_width <= 0 or max_height <= 0:
        return None

    src_h, src_w = image.shape[:2]
    if src_h == 0 or src_w == 0:
        return None

    scale = min(max_width / src_w, max_height / src_h)
    new_width = max(1, int(src_w * scale))
    new_height = max(1, int(src_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def paste_centered_image(frame, image, x, y, w, h, background_color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), background_color, -1)
    resized = resize_image_to_fit(image, w, h)
    if resized is None:
        return

    image_h, image_w = resized.shape[:2]
    offset_x = x + (w - image_w) // 2
    offset_y = y + (h - image_h) // 2
    frame[offset_y:offset_y + image_h, offset_x:offset_x + image_w] = resized


def draw_card(frame, card, active=False):
    fill_color = PANEL_BG_ALT if not active else (246, 247, 249)
    border_color = PANEL_STROKE if not active else ACCENT
    draw_panel(frame, card.x, card.y, card.w, card.h, fill_color, border_color, alpha=0.95, border_thickness=2)
    if card.image is not None:
        image_h = card.h - 34
        image_w = card.w - 20
        paste_centered_image(frame, card.image, card.x + 10, card.y + 10, image_w, image_h, fill_color)
    draw_text(frame, card.label, (card.x + 14, card.y + card.h - 28), 15, TEXT_PRIMARY if not active else (35, 35, 35), bold=True)


def draw_presenting_outline(frame, card):
    cv2.rectangle(frame, (card.x - 4, card.y - 4), (card.x + card.w + 4, card.y + card.h + 4), SUCCESS, 2)
    draw_text(frame, "LIVE", (card.x + 6, card.y + 2), 7, SUCCESS, bold=True)


def draw_hover_outline(frame, card):
    cv2.rectangle(frame, (card.x - 4, card.y - 4), (card.x + card.w + 4, card.y + card.h + 4), ACCENT, 2)


def draw_hover_progress(frame, center, radius, progress, color):
    progress = clamp(progress, 0.0, 1.0)
    cv2.circle(frame, center, radius, (55, 55, 55), 2)
    if progress <= 0:
        return
    start_angle = -90
    end_angle = int(start_angle + 360 * progress)
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 4)


def draw_zone(frame, zone, cards, hovered=False, filled=False):
    fill_color = (35, 44, 58) if filled else PANEL_BG
    if hovered:
        fill_color = (42, 52, 68)
    border_color = ACCENT if hovered else PANEL_STROKE
    draw_panel(frame, zone.x, zone.y, zone.w, zone.h, fill_color, border_color, alpha=0.96, border_thickness=3 if hovered else 2)
    draw_text(frame, zone.label, (zone.x + 18, zone.y + 12), 14, TEXT_MUTED, bold=True)


def draw_slide_preview(frame, card, zoom_scale, zone):
    panel_x = zone.x
    panel_y = zone.y
    panel_w = min(int(zone.w * zoom_scale * 0.94), frame.shape[1] - panel_x - 12)
    panel_h = min(int(zone.h * zoom_scale * 0.94), frame.shape[0] - panel_y - 18)
    panel_w = max(panel_w, 260)
    panel_h = max(panel_h, 240)

    draw_panel(frame, panel_x, panel_y, panel_w, panel_h, PANEL_BG_ALT, ACCENT, alpha=0.97, border_thickness=2)

    title = card.label[:40]
    draw_text(frame, "Presentation View", (panel_x + 18, panel_y + 10), 14, TEXT_MUTED, bold=True)
    draw_text(frame, title, (panel_x + 18, panel_y + panel_h - 32), 18, TEXT_PRIMARY, bold=True)

    image_x = panel_x + 18
    image_y = panel_y + 48
    image_w = max(panel_w - 36, 120)
    image_h = max(panel_h - 82, 120)

    if card.image is not None:
        paste_centered_image(frame, card.image, image_x, image_y, image_w, image_h, (18, 22, 30))
    else:
        cv2.rectangle(frame, (image_x, image_y), (image_x + image_w, image_y + image_h), card.color, -1)
        draw_text(frame, title, (image_x + 24, image_y + image_h // 2 - 18), 24, (30, 30, 30), bold=True)


def draw_start_screen(frame, hand_ready):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 14, 26), -1)
    frame[:] = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    draw_text(frame, "Handsome Presentation Demo", (30, 28), 28, (255, 255, 255), bold=True)
    draw_text(frame, "Raise one hand and keep it visible to begin", (30, 88), 18, (220, 220, 220), bold=False)
    draw_text(frame, "Hover on a slide to pick it up", (30, 150), 18, (160, 240, 255), bold=True)
    draw_text(frame, "Hover on the right box to drop it", (30, 184), 18, (160, 240, 255), bold=True)
    draw_text(frame, "SPACE: start   F: fullscreen   Q: quit", (30, h - 110), 18, (255, 255, 255), bold=False)

    status_color = (120, 255, 180) if hand_ready else (255, 210, 120)
    status_text = "Hand detected - ready to start" if hand_ready else "Waiting for hand"
    box_x = 30
    box_y = h - 70
    box_w = min(w - 60, 320)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + 40), (24, 30, 46), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + 40), status_color, 3)
    draw_text(frame, status_text, (box_x + 14, box_y + 4), 18, (255, 255, 255), bold=True)


def animate_card(card):
    if card.target_position is None:
        return

    target_x, target_y = card.target_position
    next_x = int(card.x + (target_x - card.x) * SNAP_SPEED)
    next_y = int(card.y + (target_y - card.y) * SNAP_SPEED)
    card.x = next_x
    card.y = next_y

    if abs(card.x - target_x) <= 2 and abs(card.y - target_y) <= 2:
        card.x = target_x
        card.y = target_y
        card.clear_target()


def load_slide_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return image


def nearest_card(cards, point, max_distance):
    best_card = None
    best_distance = max_distance
    for card in cards:
        center = card.center()
        current_distance = distance(center, point)
        if current_distance <= best_distance:
            best_distance = current_distance
            best_card = card
    return best_card


def overlap_area(card, zone):
    left = max(card.x, zone.x)
    top = max(card.y, zone.y)
    right = min(card.x + card.w, zone.x + zone.w)
    bottom = min(card.y + card.h, zone.y + zone.h)
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def best_drop_zone(card, zones):
    best_zone = None
    best_overlap = 0
    for zone in zones:
        current_overlap = overlap_area(card, zone)
        if current_overlap > best_overlap:
            best_overlap = current_overlap
            best_zone = zone
    return best_zone


def zone_under_cursor(point, zones):
    for zone in zones:
        if zone.contains(*point):
            return zone
    return None


def build_card_positions(count):
    positions = []
    for index in range(count):
        row = index // CARD_COLUMNS
        col = index % CARD_COLUMNS
        x = CARD_START_X + col * (CARD_WIDTH + CARD_GAP_X)
        y = CARD_START_Y + row * (CARD_HEIGHT + CARD_GAP_Y)
        positions.append((x, y))
    return positions


def update_card_layout(cards, positions):
    for card, (x, y) in zip(cards, positions):
        card.x = x
        card.y = y
        card.home_position = (x, y)
        card.clear_target()


def create_cards_from_folder():
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    slide_paths = sorted(
        path for path in SLIDES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    ) if SLIDES_DIR.exists() else []
    fallback_colors = [
        (77, 177, 255),
        (115, 232, 180),
        (255, 204, 102),
        (255, 140, 140),
        (180, 150, 255),
        (130, 220, 220),
    ]

    visible_count = max(3, len(slide_paths))
    positions = build_card_positions(visible_count)

    cards = []
    for index, (x, y) in enumerate(positions):
        image = None
        label = f"Slide {index + 1}"
        if index < len(slide_paths):
            image = load_slide_image(slide_paths[index])
            label = slide_paths[index].stem.replace("_", " ").replace("-", " ")

        cards.append(
            DraggableCard(
                label,
                x,
                y,
                CARD_WIDTH,
                CARD_HEIGHT,
                fallback_colors[index % len(fallback_colors)],
                image=image,
            )
        )

    return cards, len(slide_paths)


def reset_cards(cards):
    for card in cards:
        card.reset_to_home()
        card.placed_in = None


def get_visible_cards(cards, page_index):
    start = page_index * PAGE_SIZE
    end = start + PAGE_SIZE
    return cards[start:end]


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cards, loaded_slide_count = create_cards_from_folder()
    total_pages = 1
    page_index = 0
    visible_cards = cards
    update_card_layout(visible_cards, build_card_positions(len(visible_cards)))

    presentation_zone = DropZone("Presentation", 250, 95, 360, 300, (61, 111, 255))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    cursor_point = None
    dragged_card = None
    drag_offset = (0, 0)
    status_text = "Show one hand to begin"
    action_text = "Waiting"
    target_text = "-"
    active_zone_label = None
    hovered_card = None
    hovered_zone = None
    hover_pickup_since = None
    hover_drop_since = None
    drop_flash_until = 0.0
    last_drop_zone_label = None
    presented_card = None
    preview_zoom = 1.0
    previous_frame_time = time.time()
    last_hand_seen_time = previous_frame_time
    fps = 0.0
    show_start_screen = SHOW_START_SCREEN
    fullscreen_enabled = False

    print("MediaPipe Drag and Drop Demo")
    print("Move: point with your index finger")
    print("Pick up: hover over a slide")
    print("Drop: hover over the presentation box on the right")
    print(f"Slides folder: {SLIDES_DIR.resolve()}")
    print(f"Loaded slide images: {loaded_slide_count}")
    print("Press ENTER to present the hovered slide")
    print("Press 'r' to reset cards")
    print("Press 'q' to quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_landmarks_to_draw = None

        h, w, _ = frame.shape
        hand_found = False
        active_zone_label = None
        hovered_card = None
        hovered_zone = None
        sidebar_x = 16
        sidebar_y = 16
        sidebar_w = min(220, max(180, w // 3))
        sidebar_h = h - 32
        stage_x = sidebar_x + sidebar_w + 16
        stage_y = 72
        stage_w = w - stage_x - 16
        stage_h = h - stage_y - 16
        presentation_zone.x = stage_x
        presentation_zone.y = stage_y
        presentation_zone.w = max(220, stage_w)
        presentation_zone.h = max(180, stage_h)
        footer_y_1 = h - 74
        footer_y_2 = h - 46
        footer_y_3 = h - 18

        current_time = time.time()
        frame_delta = max(current_time - previous_frame_time, 1e-6)
        fps = 1.0 / frame_delta
        previous_frame_time = current_time

        if results.multi_hand_landmarks:
            hand_found = True
            last_hand_seen_time = current_time
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_landmarks_to_draw = hand_landmarks

            index_tip = hand_landmarks.landmark[8]

            target_cursor = (int(index_tip.x * w), int(index_tip.y * h))
            cursor_point = smooth_point(cursor_point, target_cursor, SMOOTHING)

            for card in reversed(visible_cards):
                if card.contains(*cursor_point):
                    hovered_card = card
                    break
            if hovered_card is None:
                hovered_card = nearest_card(visible_cards, cursor_point, GRAB_RADIUS)

            if dragged_card is None:
                if hovered_card is not None:
                    if hover_pickup_since is None or target_text != hovered_card.label:
                        hover_pickup_since = current_time
                    target_text = hovered_card.label
                    action_text = "Hover to pick"
                    status_text = f"Hover on {hovered_card.label} to pick it up"
                    if current_time - hover_pickup_since >= HOVER_PICKUP_SECONDS:
                        dragged_card = hovered_card
                        drag_offset = (cursor_point[0] - dragged_card.x, cursor_point[1] - dragged_card.y)
                        dragged_card.clear_target()
                        status_text = f"Holding {dragged_card.label}"
                        action_text = "Grabbed"
                        target_text = dragged_card.label
                        hover_pickup_since = None
                else:
                    hover_pickup_since = None
                    action_text = "Tracking"
                    status_text = "Hover over a slide to pick it up"
                    target_text = "-"
            else:
                hover_pickup_since = None

            if dragged_card is not None:
                dragged_card.x = clamp(cursor_point[0] - drag_offset[0], 0, w - dragged_card.w)
                dragged_card.y = clamp(cursor_point[1] - drag_offset[1], 90, h - dragged_card.h)
                dragged_card.placed_in = None
                dragged_card.clear_target()
                status_text = f"Dragging {dragged_card.label}"
                action_text = "Dragging"
                drop_candidate = zone_under_cursor(cursor_point, [presentation_zone])
                if drop_candidate is not None:
                    active_zone_label = drop_candidate.label
                    target_text = drop_candidate.label
                    hovered_zone = drop_candidate
                    if hover_drop_since is None:
                        hover_drop_since = current_time
                    elif current_time - hover_drop_since >= HOVER_DROP_SECONDS:
                        snap_x = drop_candidate.x + (drop_candidate.w - dragged_card.w) // 2
                        snap_y = drop_candidate.y + drop_candidate.h - dragged_card.h - 18
                        dragged_card.set_target(snap_x, snap_y)
                        dragged_card.placed_in = drop_candidate.label
                        presented_card = dragged_card
                        status_text = f"{dragged_card.label} dropped in {drop_candidate.label}"
                        action_text = "Dropped"
                        target_text = drop_candidate.label
                        drop_flash_until = current_time + DROP_FLASH_SECONDS
                        last_drop_zone_label = drop_candidate.label
                        dragged_card = None
                        hover_drop_since = None
                else:
                    hover_drop_since = None
                    hovered_zone = None
                    target_text = "None"
            else:
                hover_drop_since = None
                hovered_zone = None

            cv2.circle(frame, cursor_point, 12, (0, 255, 255), -1)

        if not hand_found and current_time - last_hand_seen_time > HAND_LOST_GRACE_SECONDS:
            cursor_point = None
            dragged_card = None
            hover_pickup_since = None
            hover_drop_since = None
            status_text = "Show one hand to begin"
            action_text = "Waiting"
            target_text = "-"

        for card in visible_cards:
            if card is not dragged_card:
                animate_card(card)

        # Draw interface after gesture updates so dragged items stay on top.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (9, 12, 18), -1)
        frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)

        draw_panel(frame, sidebar_x, sidebar_y, sidebar_w, sidebar_h, PANEL_BG, PANEL_STROKE, alpha=0.94, border_thickness=1)
        draw_text(frame, "Handsome is cooking", (sidebar_x + 16, sidebar_y + 10), 20, TEXT_PRIMARY, bold=True)
        draw_text(frame, "Slide staging demo", (sidebar_x + 16, sidebar_y + 34), 12, TEXT_MUTED, bold=False)

        draw_panel(frame, sidebar_x + 12, sidebar_y + 66, sidebar_w - 24, 96, PANEL_BG_ALT, PANEL_STROKE, alpha=0.98, border_thickness=1)
        draw_text(frame, "Session", (sidebar_x + 20, sidebar_y + 76), 14, TEXT_PRIMARY, bold=True)
        draw_metric(frame, "Action", action_text, sidebar_x + 20, sidebar_y + 102)
        draw_metric(frame, "Target", target_text, sidebar_x + 104, sidebar_y + 102)
        draw_metric(frame, "FPS", f"{fps:.1f}", sidebar_x + 20, sidebar_y + 138)

        draw_text(frame, "Slides", (sidebar_x + 16, sidebar_y + 170), 14, TEXT_MUTED, bold=True)
        draw_zone(
            frame,
            presentation_zone,
            cards,
            hovered=presentation_zone.label == active_zone_label,
            filled=presented_card is not None,
        )
        draw_text(frame, "Now Presenting", (presentation_zone.x + 12, presentation_zone.y - 28), 16, TEXT_PRIMARY, bold=True)
        if current_time < drop_flash_until and presentation_zone.label == last_drop_zone_label:
            cv2.rectangle(
                frame,
                (presentation_zone.x - 5, presentation_zone.y - 5),
                (presentation_zone.x + presentation_zone.w + 5, presentation_zone.y + presentation_zone.h + 5),
                SUCCESS,
                5,
            )
            draw_text(frame, "Dropped!", (presentation_zone.x + 20, presentation_zone.y + 16), 14, SUCCESS, bold=True)

        preview_card = dragged_card if dragged_card is not None else presented_card
        if preview_card is not None:
            draw_slide_preview(frame, preview_card, preview_zoom, presentation_zone)
        elif hovered_card is None:
            draw_text(frame, "Drop a slide here", (presentation_zone.x + 28, presentation_zone.y + 92), 18, TEXT_MUTED, bold=True)
            draw_text(frame, "Hover a slide on the left and move it here", (presentation_zone.x + 28, presentation_zone.y + 122), 12, TEXT_MUTED, bold=False)

        for card in visible_cards:
            if card is not dragged_card:
                draw_card(frame, card, active=False)
                if presented_card is card:
                    draw_presenting_outline(frame, card)

        if hovered_card is not None and hovered_card is not dragged_card:
            draw_hover_outline(frame, hovered_card)

        if dragged_card is not None:
            draw_card(frame, dragged_card, active=True)
            cv2.rectangle(frame, (dragged_card.x - 6, dragged_card.y - 6), (dragged_card.x + dragged_card.w + 6, dragged_card.y + dragged_card.h + 6), ACCENT, 2)

        completed = sum(1 for card in cards if card.placed_in is not None)
        draw_panel(frame, sidebar_x + 12, sidebar_h + sidebar_y - 78, sidebar_w - 24, 62, PANEL_BG_ALT, PANEL_STROKE, alpha=0.98, border_thickness=1)
        draw_text(frame, f"Slides  {loaded_slide_count}", (sidebar_x + 20, sidebar_h + sidebar_y - 70), 12, TEXT_MUTED, bold=False)
        draw_text(frame, f"Slides on tray  {len(visible_cards)}", (sidebar_x + 20, sidebar_h + sidebar_y - 48), 12, TEXT_MUTED, bold=False)
        draw_text(frame, f"Placed  {completed}/{len(cards)}", (sidebar_x + 110, sidebar_h + sidebar_y - 48), 12, SUCCESS, bold=True)

        if cursor_point is not None:
            if dragged_card is None and hovered_card is not None and hover_pickup_since is not None:
                pickup_progress = (current_time - hover_pickup_since) / HOVER_PICKUP_SECONDS
                draw_hover_progress(frame, cursor_point, 22, pickup_progress, (0, 255, 255))
            elif dragged_card is not None and hovered_zone is not None and hover_drop_since is not None:
                drop_progress = (current_time - hover_drop_since) / HOVER_DROP_SECONDS
                draw_hover_progress(frame, cursor_point, 22, drop_progress, (120, 255, 180))

        if hand_landmarks_to_draw is not None:
            mp_draw.draw_landmarks(frame, hand_landmarks_to_draw, mp_hands.HAND_CONNECTIONS)

        if show_start_screen:
            draw_start_screen(frame, hand_found)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            fullscreen_enabled = not fullscreen_enabled
            mode = cv2.WINDOW_FULLSCREEN if fullscreen_enabled else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)
        if key == ord(" "):
            show_start_screen = False
        if key == 13:
            if not show_start_screen and hovered_card is not None:
                presented_card = hovered_card
                presented_card.placed_in = presentation_zone.label
                status_text = f"{presented_card.label} presented"
                action_text = "Keyboard present"
                target_text = presentation_zone.label
                drop_flash_until = current_time + DROP_FLASH_SECONDS
                last_drop_zone_label = presentation_zone.label
        if key == ord("z"):
            preview_zoom = min(1.35, preview_zoom + 0.05)
        if key == ord("x"):
            preview_zoom = max(0.75, preview_zoom - 0.05)
        if key == ord("r"):
            reset_cards(cards)
            visible_cards = cards
            update_card_layout(visible_cards, build_card_positions(len(visible_cards)))
            dragged_card = None
            presented_card = None
            hovered_card = None
            hover_pickup_since = None
            hover_drop_since = None
            status_text = "Cards reset"
            action_text = "Reset"
            target_text = "-"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
