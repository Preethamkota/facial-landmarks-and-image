from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "public" / "train"
DEFAULT_FACEMESH_OUTPUT_DIR = ROOT_DIR / "public" / "train_facemesh"
DEFAULT_LANDMARKS_OUTPUT_DIR = ROOT_DIR / "public" / "train_landmarks"
MODEL_PATH = ROOT_DIR / "models" / "face_landmarker.task"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_FACEMESH_SIZE = 224


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def slugify_frame_id(label: str, stem: str) -> str:
    safe_label = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()
    safe_stem = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return f"{safe_label}_{safe_stem}" if safe_stem else safe_label


def build_output_key(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    return f"{image_path.stem}__{suffix}"


def ensure_rgb_image(image) -> "cv2.typing.MatLike":
    if image is None:
        return None

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_mesh(face_landmarks, output_size: int = DEFAULT_FACEMESH_SIZE) -> "cv2.typing.MatLike":
    canvas = 255 * (cv2.UMat(output_size, output_size, cv2.CV_8UC3).get() * 0 + 1)

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


def iter_images(input_dir: Path):
    for label_dir in sorted(input_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield label_dir.name, image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate facemesh images and landmarks JSON for a labeled image dataset.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Dataset root with one subfolder per label.",
    )
    parser.add_argument(
        "--facemesh-output-dir",
        type=Path,
        default=DEFAULT_FACEMESH_OUTPUT_DIR,
        help="Where facemesh preview images should be written.",
    )
    parser.add_argument(
        "--landmarks-output-dir",
        type=Path,
        default=DEFAULT_LANDMARKS_OUTPUT_DIR,
        help="Where landmarks JSON files should be written.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete and recreate the facemesh and landmark output folders before processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    facemesh_output_dir = args.facemesh_output_dir.resolve()
    landmarks_output_dir = args.landmarks_output_dir.resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")

    if args.clean_output:
        reset_dir(facemesh_output_dir)
        reset_dir(landmarks_output_dir)
    else:
        ensure_dir(facemesh_output_dir)
        ensure_dir(landmarks_output_dir)

    processed = 0
    skipped = 0

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )

    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        for label, image_path in iter_images(input_dir):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"Skipped unreadable image: {image_path}")
                skipped += 1
                continue
            image_rgb = ensure_rgb_image(image_bgr)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = face_landmarker.detect(mp_image)

            if not results.face_landmarks:
                print(f"Skipped no-face image: {image_path}")
                skipped += 1
                continue

            face_landmarks = results.face_landmarks[0]
            output_key = build_output_key(image_path)
            frame_id = slugify_frame_id(label, output_key)

            mesh_label_dir = facemesh_output_dir / label
            landmarks_label_dir = landmarks_output_dir / label
            ensure_dir(mesh_label_dir)
            ensure_dir(landmarks_label_dir)

            mesh_image = draw_mesh(face_landmarks)
            mesh_output_path = mesh_label_dir / f"{output_key}_mesh.png"
            json_output_path = landmarks_label_dir / f"{output_key}.json"

            cv2.imwrite(str(mesh_output_path), mesh_image)
            write_landmarks_json(json_output_path, frame_id, label, face_landmarks)

            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed} images...")

    print(
        f"Finished. Input={input_dir}, Processed={processed}, Skipped={skipped}, "
        f"FacemeshOutput={facemesh_output_dir}, LandmarksOutput={landmarks_output_dir}"
    )

def extract_landmarks_from_frame(frame, face_landmarker):
    import mediapipe as mp

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    results = face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None

    face_landmarks = results.face_landmarks[0]

    # Convert to list
    landmarks = [
        [lm.x, lm.y, lm.z]
        for lm in face_landmarks
    ]

    return landmarks

if __name__ == "__main__":
    main()
