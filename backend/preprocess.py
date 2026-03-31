import math

def preprocess(landmarks):
    # -------- CENTER --------
    ref = landmarks[0]
    centered = [[x - ref[0], y - ref[1]] for x, y, z in landmarks]

    # -------- SCALE --------
    left_eye = centered[33]
    right_eye = centered[263]

    scale = math.sqrt(
        (left_eye[0] - right_eye[0])**2 +
        (left_eye[1] - right_eye[1])**2
    ) + 1e-8

    scaled = [[x/scale, y/scale] for x, y in centered]

    # -------- FLATTEN --------
    flat = []
    for point in scaled:
        flat.extend(point)

    x = torch.tensor(flat, dtype=torch.float32)

    # -------- NORMALIZATION --------
    x = (x - x.mean()) / (x.std() + 1e-8)

    return x.unsqueeze(0)  # add batch dim
