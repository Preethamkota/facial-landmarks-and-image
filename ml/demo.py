"""
demo.py  —  Privacy-safe expression data collection
=====================================================
Run:  python demo.py --subject S001 --video V001

What it does every 3 seconds:
  1. Captures a frame from webcam
  2. Extracts 478 facial landmarks using face_landmarker.task  (no face stored)
  3. Predicts expression using expression.onnx
  4. Stores ONLY landmarks + expression in MongoDB  (frame is discarded)

Install:
  pip install opencv-python mediapipe onnxruntime pymongo python-dotenv
"""

import cv2
import time
import math
import argparse
import numpy as np
import onnxruntime as ort
from datetime import datetime, timezone
from pymongo import MongoClient
from pathlib import Path

# MediaPipe Python API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths — resolved relative to this script file ────────
# Works correctly whether you run from ml/ or from the project root
HERE = Path(__file__).parent.resolve()   # always = /path/to/ml/

FACE_LANDMARKER_PATH = str(HERE.parent / "my_react_app" / "models" / "face_landmarker.task")
ONNX_MODEL_PATH      = str(HERE / "expression.onnx")
# ── Config ────────────────────────────────────────────────
MONGO_URI            = "mongodb://localhost:27017"   # change to Atlas URI for clinic
MONGO_DB             = "therapy_db"
MONGO_COLLECTION     = "expressions"
CAPTURE_INTERVAL_SEC = 3
CONFIDENCE_THRESHOLD = 0.35
MODEL_VERSION        = "mlp_v3"

LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

LANDMARK_GROUPS = {
    "left_eye":   [33, 160, 158, 133, 153, 144],
    "right_eye":  [362, 385, 387, 263, 373, 380],
    "mouth":      [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "left_brow":  [70, 63, 105, 66, 107],
    "right_brow": [336, 296, 334, 293, 300],
    "nose":       [1, 2, 4, 5, 6],
}

# Expression → color (BGR for OpenCV)
EXPRESSION_COLOR = {
    "happy":    (86, 158, 29),   # green
    "neutral":  (128, 135, 136), # gray
    "sad":      (186, 116, 55),  # blue
    "angry":    (48, 90, 216),   # red
    "fear":     (180, 119, 127), # purple
    "surprise": (23, 117, 186),  # orange
}


# ═══════════════════════════════════════════════════════════
# PREPROCESSING  (identical to training code)
# ═══════════════════════════════════════════════════════════
def group_centroid(scaled, indices):
    xs = [scaled[i][0] for i in indices]
    ys = [scaled[i][1] for i in indices]
    return sum(xs)/len(xs), sum(ys)/len(ys)


def preprocess(landmarks):
    """landmarks: list of NormalizedLandmark from MediaPipe"""
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks]

    ref = pts[1]
    centered = [(x - ref[0], y - ref[1]) for x, y, z in pts]

    le, re = centered[33], centered[263]
    scale = math.sqrt((le[0]-re[0])**2 + (le[1]-re[1])**2) + 1e-8
    scaled = [(x/scale, y/scale) for x, y in centered]

    features = []
    for x, y in scaled:
        features.extend([x, y, math.sqrt(x*x+y*y), math.atan2(y, x)])

    centroids   = {n: group_centroid(scaled, idx) for n, idx in LANDMARK_GROUPS.items()}
    group_names = list(LANDMARK_GROUPS.keys())
    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            ax, ay = centroids[group_names[i]]
            bx, by = centroids[group_names[j]]
            features.append(math.sqrt((ax-bx)**2+(ay-by)**2))

    def ear(pts_idx):
        p = [scaled[i] for i in pts_idx]
        v1 = math.sqrt((p[1][0]-p[5][0])**2+(p[1][1]-p[5][1])**2)
        v2 = math.sqrt((p[2][0]-p[4][0])**2+(p[2][1]-p[4][1])**2)
        h  = math.sqrt((p[0][0]-p[3][0])**2+(p[0][1]-p[3][1])**2)+1e-8
        return (v1+v2)/(2*h)

    def mar():
        top, bot = scaled[13], scaled[14]
        l, r     = scaled[61], scaled[291]
        v = math.sqrt((top[0]-bot[0])**2+(top[1]-bot[1])**2)
        h = math.sqrt((l[0]-r[0])**2+(l[1]-r[1])**2)+1e-8
        return v/h

    le_ear = ear(LANDMARK_GROUPS["left_eye"])
    re_ear = ear(LANDMARK_GROUPS["right_eye"])
    features.extend([le_ear, re_ear, (le_ear+re_ear)/2, mar()])

    lbx,lby = centroids["left_brow"];  lex,ley = centroids["left_eye"]
    rbx,rby = centroids["right_brow"]; rex,rey = centroids["right_eye"]
    lr = math.sqrt((lbx-lex)**2+(lby-ley)**2)
    rr = math.sqrt((rbx-rex)**2+(rby-rey)**2)
    features.extend([lr, rr, (lr+rr)/2])

    arr  = np.array(features, dtype=np.float32)
    arr  = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr.reshape(1, -1)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ═══════════════════════════════════════════════════════════
# OVERLAY DRAWING  (on the preview window only — not stored)
# ═══════════════════════════════════════════════════════════
def draw_landmarks(frame, landmarks, color=(0, 255, 100)):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 1, color, -1)


def draw_hud(frame, expression, confidence, countdown, frame_count, subject_id, stored):
    h, w = frame.shape[:2]
    color = EXPRESSION_COLOR.get(expression, (200, 200, 200))

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Expression label
    cv2.putText(frame, expression.upper() if expression else "...",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Confidence
    if confidence:
        conf_text = f"{confidence*100:.0f}%"
        cv2.putText(frame, conf_text,
                    (200, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 1, cv2.LINE_AA)

    # Countdown ring (bottom right)
    cx, cy, r = w - 50, h - 50, 28
    angle = int(360 * (1 - countdown / CAPTURE_INTERVAL_SEC))
    cv2.circle(frame, (cx, cy), r, (60, 60, 60), 3)
    if angle > 0:
        cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, angle, (86, 158, 29), 3)
    cv2.putText(frame, str(max(0, math.ceil(countdown))),
                (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)

    # Bottom bar: subject, frame count, saved indicator
    cv2.rectangle(frame, (0, h-32), (w, h), (20, 20, 20), -1)
    info = f"Subject: {subject_id}   Frames stored: {frame_count}"
    cv2.putText(frame, info, (12, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1, cv2.LINE_AA)

    # "SAVED" flash
    if stored:
        cv2.putText(frame, "SAVED", (w - 90, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (86, 200, 86), 2, cv2.LINE_AA)

    # Privacy reminder
    cv2.putText(frame, "NO IMAGE STORED  |  LANDMARKS ONLY",
                (w//2 - 180, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (80, 120, 80), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main(subject_id: str, video_id: str):
    # ── Validate paths before loading ────────────────────
    if not Path(FACE_LANDMARKER_PATH).exists():
        print(f"[error] face_landmarker.task not found at:\n        {FACE_LANDMARKER_PATH}")
        print("        Make sure my_react_app/models/face_landmarker.task exists.")
        return
    if not Path(ONNX_MODEL_PATH).exists():
        print(f"[error] expression.onnx not found at:\n        {ONNX_MODEL_PATH}")
        print("        Make sure ml/expression.onnx exists (run mmlp.py first).")
        return

    # ── Load models ──────────────────────────────────────
    print("[init] Loading face landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    print("[init] Face landmarker ready.")

    print("[init] Loading ONNX model...")
    ort_session  = ort.InferenceSession(ONNX_MODEL_PATH,
                                         providers=["CPUExecutionProvider"])
    input_name   = ort_session.get_inputs()[0].name
    print("[init] ONNX model ready.")

    print("[init] Connecting to MongoDB...")
    mongo_col = MongoClient(MONGO_URI)[MONGO_DB][MONGO_COLLECTION]
    print("[init] MongoDB ready.")

    # ── Open webcam ───────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Could not open webcam.")
        return

    session_id  = f"{subject_id}_{int(time.time())}"
    frame_count = 0          # how many frames stored
    last_capture = time.time() - CAPTURE_INTERVAL_SEC  # capture immediately at start

    expression = None
    confidence = None
    stored_flash = 0   # countdown for "SAVED" indicator

    print(f"\n[ready] Session: {session_id}")
    print(f"        Subject: {subject_id}  |  Video: {video_id}")
    print("        Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        now   = time.time()
        countdown = CAPTURE_INTERVAL_SEC - (now - last_capture)
        just_stored = False

        # ── Capture + process every N seconds ────────────
        if countdown <= 0:
            last_capture = now

            # Step 1: Landmarks
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            if result.face_landmarks and len(result.face_landmarks) > 0:
                raw_lms = result.face_landmarks[0]   # 478 landmarks

                # Draw landmarks on preview (not stored)
                draw_landmarks(frame, raw_lms)

                # Step 2: Preprocess + ONNX inference
                features = preprocess(raw_lms)
                logits   = ort_session.run(None, {input_name: features})[0][0]
                probs    = softmax(logits)
                top_idx  = int(probs.argmax())
                expression = LABELS[top_idx]
                confidence = float(probs[top_idx])
                all_scores = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}

                print(f"[frame {frame_count+1}] {expression:>8}  conf={confidence:.2f}  ", end="")

                # Step 3: Store to MongoDB (NO image — landmarks only)
                if confidence >= CONFIDENCE_THRESHOLD:
                    doc = {
                        # Identity
                        "subjectId":   subject_id,
                        "videoId":     video_id,
                        "sessionId":   session_id,

                        # Timing
                        "timestamp":   datetime.now(timezone.utc),
                        "frameIndex":  frame_count,

                        # Landmarks ONLY — privacy safe, no face image
                        "landmarks": [
                            {"x": round(lm.x, 6), "y": round(lm.y, 6)}
                            for lm in raw_lms
                        ],

                        # Prediction
                        "expression":  expression,
                        "confidence":  round(confidence, 4),
                        "allScores":   all_scores,

                        # Metadata
                        "modelVersion":      MODEL_VERSION,
                        "captureIntervalSec": CAPTURE_INTERVAL_SEC,
                    }
                    mongo_col.insert_one(doc)
                    frame_count += 1
                    just_stored  = True
                    stored_flash = 1.5   # show "SAVED" for 1.5 seconds
                    print(f"→ STORED  (total: {frame_count})")
                else:
                    print(f"→ skipped (low confidence)")
            else:
                expression = None
                confidence = None
                print(f"[frame] No face detected — skipping")

        # ── Draw HUD on preview frame ─────────────────────
        if stored_flash > 0:
            stored_flash -= 1 / 30   # assumes ~30fps
        show_saved = stored_flash > 0

        draw_hud(frame, expression or "no face", confidence,
                 max(0, countdown), frame_count, subject_id, show_saved)

        cv2.imshow("Expression pipeline  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ───────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print(f"\n[done] Session complete.")
    print(f"       Subject: {subject_id}  |  Frames stored: {frame_count}")
    print(f"       Collection: {MONGO_DB}.{MONGO_COLLECTION}")
    print(f"       Session ID: {session_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy-safe expression data collection")
    parser.add_argument("--subject", required=True, help="Subject ID  e.g. S001")
    parser.add_argument("--video",   required=True, help="Video/session ID  e.g. V001")
    args = parser.parse_args()

    main(subject_id=args.subject, video_id=args.video)