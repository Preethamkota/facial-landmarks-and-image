import time
import cv2 as cv

capture = cv.VideoCapture(0)

interval = 3
last_time=0

while True:
    _,frame = capture.read()

    cv.imshow("live recording",frame)

    cur_time = time.time()

    if cur_time-last_time >= interval:
        cv.imwrite(f"frames/frame_{cur_time}.jpg",frame)
        last_time=cur_time

    if cv.waitKey(1) & 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()