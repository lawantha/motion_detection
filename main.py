import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("trained_model.h5")

# cascade file for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# get video from web cam
cap = cv2.VideoCapture(1)

#loop for capture all frames
while True:
    # captures frame and returns boolean value and captured image start
    retv, test_img = cap.read()

    #check that the frame is empty
    if (test_img is None):
        print("Received empty frame. Exiting")
        cap.release()
        #capture using cv2.CAP_DSHOW (in windows7 opencv could not display video while using third party camera)
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        print(cap)
        retv, test_img = cap.read()

    cv2.normalize(test_img, test_img, 0, 255, cv2.NORM_MINMAX)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

    #OpenCV Video I/O API Backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # capturing image stops

    # Fliping the image
    flip_img = cv2.flip(test_img, 1).resize(48,48)

    #Converting the input frame to grayscale
    gray_img = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(flip_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        #roi_gray = cv2.resize(roi_gray, (224, 224))

        #reshape_img = roi_gray.reshape(48,48,1)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        #emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion = ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'natural')
        predicted_emotion = emotion[max_index]

        cv2.putText(flip_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(flip_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', gray_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows