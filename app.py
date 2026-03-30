import cv2
import numpy as np
import os
import sys
import mediapipe as mp
import threading

# ─── Install keyboard if missing ──────────────────────────────────────────────
try:
    import keyboard
except ImportError:
    os.system("pip install keyboard")
    import keyboard

# ─── MediaPipe Setup ───────────────────────────────────────────────────────────
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
FOREHEAD        = 10
CHIN            = 152
NOSE_TIP        = 1

ACCESSORIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accessories")

# ─── Global state (controlled by keyboard hooks) ───────────────────────────────
state = {
    "idx": 1,
    "landmarks": False,
    "quit": False
}

def on_d(e):
    state["idx"] = (state["idx"] + 1) % 7
    print(f">>> Switched to: {ACCESSORIES[state['idx']]['name']}")

def on_a(e):
    state["idx"] = (state["idx"] - 1) % 7
    print(f">>> Switched to: {ACCESSORIES[state['idx']]['name']}")

def on_l(e):
    state["landmarks"] = not state["landmarks"]
    print(f">>> Landmarks: {'ON' if state['landmarks'] else 'OFF'}")

def on_q(e):
    state["quit"] = True
    print(">>> Quitting...")


def load_accessory(name):
    path = os.path.join(ACCESSORIES_DIR, name)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARNING] Could not load {path}")
    return img


def overlay_png(background, overlay, x, y, w, h):
    if overlay is None or w <= 0 or h <= 0:
        return background
    resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    bh, bw  = background.shape[:2]
    ox1 = max(0, -x);     oy1 = max(0, -y)
    cx1 = max(0,  x);     cy1 = max(0,  y)
    cx2 = min(bw, x + w); cy2 = min(bh, y + h)
    if cx2 <= cx1 or cy2 <= cy1:
        return background
    roi   = background[cy1:cy2, cx1:cx2]
    patch = resized[oy1:oy1+(cy2-cy1), ox1:ox1+(cx2-cx1)]
    if patch.shape[2] == 4:
        alpha   = patch[:, :, 3:4] / 255.0
        color   = patch[:, :, :3]
        blended = (color * alpha + roi * (1 - alpha)).astype(np.uint8)
        background[cy1:cy2, cx1:cx2] = blended
    else:
        background[cy1:cy2, cx1:cx2] = patch[:, :, :3]
    return background


def get_point(landmarks, idx, w, h):
    lm = landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


def apply_glasses(frame, landmarks, img):
    h, w   = frame.shape[:2]
    lx, ly = get_point(landmarks, LEFT_EYE_OUTER,  w, h)
    rx, ry = get_point(landmarks, RIGHT_EYE_OUTER, w, h)
    ew     = int(abs(rx - lx) * 1.55)
    eh     = int(ew * (img.shape[0] / img.shape[1]))
    cx, cy = (lx + rx) // 2, (ly + ry) // 2
    return overlay_png(frame, img, cx - ew // 2, cy - eh // 2, ew, eh)


def apply_hat(frame, landmarks, img):
    h, w   = frame.shape[:2]
    lx, _  = get_point(landmarks, LEFT_EYE_OUTER,  w, h)
    rx, _  = get_point(landmarks, RIGHT_EYE_OUTER, w, h)
    fx, fy = get_point(landmarks, FOREHEAD, w, h)
    hw     = int(abs(rx - lx) * 2.2)
    hh     = int(hw * (img.shape[0] / img.shape[1]))
    return overlay_png(frame, img, fx - hw // 2, fy - hh + int(hh * 0.15), hw, hh)


def apply_mask(frame, landmarks, img):
    h, w   = frame.shape[:2]
    lx, _  = get_point(landmarks, LEFT_EYE_OUTER,  w, h)
    rx, _  = get_point(landmarks, RIGHT_EYE_OUTER, w, h)
    _, ny  = get_point(landmarks, NOSE_TIP, w, h)
    _, cy  = get_point(landmarks, CHIN,     w, h)
    mw     = int(abs(rx - lx) * 1.6)
    mh     = int(abs(cy - ny) * 1.4)
    cx     = (lx + rx) // 2
    return overlay_png(frame, img, cx - mw // 2, ny, mw, mh)


def create_placeholder_accessories():
    os.makedirs(ACCESSORIES_DIR, exist_ok=True)

    def blank(h, w):
        return np.zeros((h, w, 4), dtype=np.uint8)

    sg = blank(120, 400)
    cv2.ellipse(sg, (100, 60), (80, 50), 0, 0, 360, (20, 20, 20, 230), -1)
    cv2.ellipse(sg, (100, 60), (80, 50), 0, 0, 360, (0,  0,  0, 255),  3)
    cv2.ellipse(sg, (300, 60), (80, 50), 0, 0, 360, (20, 20, 20, 230), -1)
    cv2.ellipse(sg, (300, 60), (80, 50), 0, 0, 360, (0,  0,  0, 255),  3)
    cv2.line(sg, (180, 60), (220, 60), (0, 0, 0, 255), 5)
    cv2.imwrite(os.path.join(ACCESSORIES_DIR, "sunglasses.png"), sg)

    rg = blank(120, 400)
    cv2.ellipse(rg, (100, 60), (70, 55), 0, 0, 360, (80, 40, 10, 200), 5)
    cv2.ellipse(rg, (300, 60), (70, 55), 0, 0, 360, (80, 40, 10, 200), 5)
    cv2.line(rg, (170, 60), (230, 60), (80, 40, 10, 255), 5)
    cv2.imwrite(os.path.join(ACCESSORIES_DIR, "round_glasses.png"), rg)

    ph = blank(300, 200)
    pts = np.array([[100, 10], [15, 280], [185, 280]], np.int32)
    cv2.fillPoly(ph, [pts], (0, 0, 200, 230))
    for cx_, cy_, r in [(60, 150, 12), (130, 100, 10), (80, 220, 8)]:
        cv2.circle(ph, (cx_, cy_), r, (255, 255, 0, 255), -1)
    cv2.imwrite(os.path.join(ACCESSORIES_DIR, "party_hat.png"), ph)

    ch = blank(250, 400)
    cv2.ellipse(ch, (200, 180), (190, 50), 0, 0, 360, (80, 50, 20, 230), -1)
    pts2 = np.array([[80, 180], [120, 50], [280, 50], [320, 180]], np.int32)
    cv2.fillPoly(ch, [pts2], (100, 65, 30, 240))
    cv2.imwrite(os.path.join(ACCESSORIES_DIR, "cowboy_hat.png"), ch)

    mm = blank(160, 300)
    cv2.ellipse(mm, (150, 80), (140, 70), 0, 0, 360, (200, 200, 200, 220), -1)
    for y_line in [50, 80, 110]:
        cv2.line(mm, (20, y_line), (280, y_line), (170, 170, 170, 180), 2)
    cv2.imwrite(os.path.join(ACCESSORIES_DIR, "medical_mask.png"), mm)

    nm = blank(160, 300)
    cv2.rectangle(nm, (0, 0), (300, 160), (20, 20, 20, 230), -1)
    cv2.imwrite(os.path.join(ACCESSORIES_DIR, "ninja_mask.png"), nm)

    print("[INFO] Accessories created.")


ACCESSORIES = [
    {"name": "None",          "file": None,                "type": None},
    {"name": "Sunglasses",    "file": "sunglasses.png",    "type": "glasses"},
    {"name": "Round Glasses", "file": "round_glasses.png", "type": "glasses"},
    {"name": "Party Hat",     "file": "party_hat.png",     "type": "hat"},
    {"name": "Cowboy Hat",    "file": "cowboy_hat.png",    "type": "hat"},
    {"name": "Medical Mask",  "file": "medical_mask.png",  "type": "mask"},
    {"name": "Ninja Mask",    "file": "ninja_mask.png",    "type": "mask"},
]

APPLY_FN = {
    "glasses": apply_glasses,
    "hat":     apply_hat,
    "mask":    apply_mask,
}


def draw_ui(frame, idx):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame,
                f"  [{idx+1}/{len(ACCESSORIES)}] {ACCESSORIES[idx]['name']}   "
                f"A = prev   D = next   L = landmarks   Q = quit",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def main():
    if not os.path.isdir(ACCESSORIES_DIR) or not os.listdir(ACCESSORIES_DIR):
        create_placeholder_accessories()

    loaded = [load_accessory(a["file"]) if a["file"] else None for a in ACCESSORIES]

    # Register global keyboard hooks (work regardless of window focus)
    keyboard.on_press_key("d", on_d)
    keyboard.on_press_key("a", on_a)
    keyboard.on_press_key("l", on_l)
    keyboard.on_press_key("q", on_q)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Try changing VideoCapture(0) to VideoCapture(1).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n>>> App running! Press A / D to switch accessories (works even from terminal)")
    print(">>> Press Q to quit\n")

    while not state["quit"]:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if result.multi_face_landmarks:
            for face_lms in result.multi_face_landmarks:
                if state["landmarks"]:
                    fh, fw = frame.shape[:2]
                    for lm in face_lms.landmark:
                        px, py = int(lm.x * fw), int(lm.y * fh)
                        cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

                acc = ACCESSORIES[state["idx"]]
                if acc["type"] and loaded[state["idx"]] is not None:
                    frame = APPLY_FN[acc["type"]](frame, face_lms, loaded[state["idx"]])

        draw_ui(frame, state["idx"])
        cv2.imshow("Virtual Try-On", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    keyboard.unhook_all()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
