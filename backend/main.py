from capture.camera import FrameCapture
from worker.emotion_worker import  start_worker

capture = FrameCapture()
capture.start()

start_worker()