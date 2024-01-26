from __future__ import division
import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import imutils
from imutils import face_utils
import RPi.GPIO as GPIO

GPIO.setwarnings(False)
# Set up GPIO
buzzer_pin = 17  # Change this to the GPIO pin you are using
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

print('Code Start')



def buzzer_beep():
    print('buzz beep')
    GPIO.output(buzzer_pin, GPIO.HIGH)
    time.sleep(1)  # Adjust the duration of the beep as needed
    GPIO.output(buzzer_pin, GPIO.LOW)



def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36, 48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize camera and face detection
camera = cv2.VideoCapture(0)
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize drowsiness detection variables
total = 0
start_time = None
drowsiness_threshold = 3  # Threshold for drowsiness detection in seconds
drowsy = False  # Flag to indicate drowsiness


buzzer_beep()
time.sleep(1)

while True:
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from the camera. Check the camera index in cv2.VideoCapture(0).\n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

    dets = detector(frame_resized, 1)


    if drowsy == True:
        GPIO.output(buzzer_pin, GPIO.HIGH)
        print('buzz on')
    if drowsy == False:
        GPIO.output(buzzer_pin, GPIO.LOW)
        print('buzz off')
        
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            if ear > 0.25:
                if start_time is not None:
                    start_time = None
                drowsy = False
                print('Eyes Opne')
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if start_time is None:
                    start_time = time.time()
                
                elapsed_time = time.time() - start_time

                if elapsed_time >= drowsiness_threshold:
                    drowsy = True
                    print('Eyes Close, Drowsiness Detected')
                    cv2.putText(frame, f"Eyes Closed ({round(elapsed_time)}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Drowsiness Detected", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)
                else:
                    drowsy = False
                    print('Eyes Closed')
                    cv2.putText(frame, f"Eyes Closed ({round(elapsed_time)}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                for (x, y) in shape:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        GPIO.cleanup()  # Clean up GPIO on program exit
        cv2.destroyAllWindows()
        camera.release()
        break
