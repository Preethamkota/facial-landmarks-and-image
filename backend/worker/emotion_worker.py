import time
import threading
from mediapipe_module.face_mesh import get_landmarks
from model.interface import predict
from db.mongo import collection
from datetime import datetime

def save(person_id,frame_no,emotion,landmarks):
    collection.insert_one({
        "person_id":person_id,
        "frame_no":frame_no,
        "emotion":emotion,
        "landmarks": [[x,y] for x,y,_ in landmarks],
        "timestamp":datetime.utcnow()
    })

def worker(capture,person_id):
    frame_no=0
    while True:
        time.sleep(3)

        frame=capture.get_frame()
        if frame in None:
            continue

        landmarks = get_landmarks(frame)
        if landmarks is None:
            continue

        emotion = predict(landmarks)
        frames+=1

        save(person_id,frame_no,emotion,landmarks)

def start_worker(capture,person_id):
    threading.Thread(
        target=worker,
        args=(capture,person_id),
        daemon=True
    ).start
