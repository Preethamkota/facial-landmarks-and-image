import time
import cv2 as cv
import torch
import numpy as np
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python import vision,BaseOptions

from ml.model import MLP
from my_react_app.scripts.generate_facemesh_dataset import extract_landmarks_from_frame
from preprocess import preprocess

MODEL_PATH = 'models/face_landmarker.task'
MLP_PATH='ml/best_model.pth'

NUM_CLASSES=4
INPUT_SIZE=936

labels=["anger","fear","joy","Natural"]

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

model = MLP(input_size=INPUT_SIZE,num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MLP_PATH,map_location="cpu"))
model.eval()

history = deque(maxlen=5)

def smooth_prediction(pred):
    history.append(pred)
    return max(set(history), key=history.count)

cap = cv.VideoCapture(0)

last_time=0
interval=3
text="None"
while True:
    ret,frame = cap.read()
    if not ret:
        break

    cur_time=time.time()
    if cur_time - last_time >= interval:
        landmarks = extract_landmarks_from_frame(frame,face_landmarker)
        last_time=cur_time
        if landmarks is not None:
            x=preprocess(landmarks)
            if x is not None:
                with torch.no_grad():
                    output=model(x)
                    pred=torch.argmax(output,dim=1).item()
                text = labels[pred]
            else:
                text="scale error"
        else:
            text="no face detected"
    
    cv.putText(frame,text,(30,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.imshow("emotion detected",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()