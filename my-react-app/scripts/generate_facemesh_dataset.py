from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "public" / "train"
FACEMESH_OUTPUT_DIR = ROOT_DIR / "public" / "train_facemesh"
LANDMARKS_OUTPUT_DIR = ROOT_DIR / "public" / "train_landmarks"
MODEL_PATH = ROOT_DIR / "models" / "face_landmarker.task"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify_frame_id(label: str, stem: str) -> str:
    safe_label = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()
    safe_stem = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return f"{safe_label}_{safe_stem}" if safe_stem else safe_label


def build_output_key(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    return f"{image_path.stem}__{suffix}"


def draw_mesh(image_shape: tuple[int, int, int], face_landmarks) -> "cv2.typing.MatLike":
    height, width = image_shape[:2]
    canvas = 255 * (cv2.UMat(height, width, cv2.CV_8UC3).get() * 0 + 1)

    drawing_spec = vision.drawing_utils.DrawingSpec(
        color=(110, 110, 110),
        thickness=1,
        circle_radius=1,
    )

    vision.drawing_utils.draw_landmarks(
        image=canvas,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec,
    )

    return canvas


def write_landmarks_json(
    output_path: Path,
    frame_id: str,
    label: str,
    face_landmarks,
) -> None:
    payload = {
        "frame_id": frame_id,
        "label": label,
        "landmarks": [
            [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)] for lm in face_landmarks
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def iter_images():
    for label_dir in sorted(INPUT_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield label_dir.name, image_path


def main() -> None:
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")

    ensure_dir(FACEMESH_OUTPUT_DIR)
    ensure_dir(LANDMARKS_OUTPUT_DIR)

    processed = 0
    skipped = 0

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )

    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        for label, image_path in iter_images():
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"Skipped unreadable image: {image_path}")
                skipped += 1
                continue

            mp_image = mp.Image.create_from_file(str(image_path))
            results = face_landmarker.detect(mp_image)

            if not results.face_landmarks:
                print(f"Skipped no-face image: {image_path}")
                skipped += 1
                continue

            face_landmarks = results.face_landmarks[0]
            output_key = build_output_key(image_path)
            frame_id = slugify_frame_id(label, output_key)

            mesh_label_dir = FACEMESH_OUTPUT_DIR / label
            landmarks_label_dir = LANDMARKS_OUTPUT_DIR / label
            ensure_dir(mesh_label_dir)
            ensure_dir(landmarks_label_dir)

            mesh_image = draw_mesh(image_bgr.shape, face_landmarks)
            mesh_output_path = mesh_label_dir / f"{output_key}_mesh.png"
            json_output_path = landmarks_label_dir / f"{output_key}.json"

            cv2.imwrite(str(mesh_output_path), mesh_image)
            write_landmarks_json(json_output_path, frame_id, label, face_landmarks)

            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed} images...")

    print(f"Finished. Processed={processed}, Skipped={skipped}")


if __name__ == "__main__":
    main()
