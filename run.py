import math
import sys
import time
import torch
import numpy as np
import cv2

# mediapipe imports
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# global declarations
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5)

MAX_PEOPLE = 5


def get_eye_pos(detection, height, width):
    xmin = (detection.location_data.relative_bounding_box.xmin)
    ymin = (detection.location_data.relative_bounding_box.ymin)
    widthF =(detection.location_data.relative_bounding_box.width)
    heightF = (detection.location_data.relative_bounding_box.height)
    

    x0 = math.floor(max(0, xmin - widthF*0.5)*width)
    x1 = math.floor(min(width, x0 + (widthF+widthF)*width))
    y0 = math.floor(max(0, ymin - heightF*0.5)*height)
    y1 = math.floor(min(height, y0+(heightF+ heightF)*height))

    eyePos = [(x0+x1)/2.0, (0.65*y0+0.35*y1)]
    return eyePos

def process_image(image: np.array) ->np.array:
    '''
    Take the image, compute the gaze and return the image with gaze drawn back
    '''
    h, w, c  = image.shape
    results = face_detection.process(image)
    if results.detections:
        for i, detection in enumerate(results.detections):
            eyePos = get_eye_pos(detection, h, w)


    return image


def main():
    
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        
        success, image = cap.read()
       
        if not success:
            print("Image Not Read!")
            continue

        image = process_image(image)        
        
        cv2.imshow("X", image)
        if cv2.waitKey(1) & 0xFF == 27:
           cap.release()
           cv2.destroyAllWindows()
           break
        
        end_time = time.time()

        
    


if __name__ == '__main__':
    main()