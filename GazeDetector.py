import math
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model import GazeLSTM

class GazeDetector:
    def __init__(self, MAX_PEOPLE=5) -> None:
        self.FLUSH_AFTER_N_FRAMES = 7
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # transforms for the image
        image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([
                transforms.Resize((224,224)),transforms.ToTensor(),image_normalize,
        ])

        # define the gaze model
        model = GazeLSTM()
        self.model = torch.nn.parallel(model)
        checkpoint = torch.load('next_model/model_best_Gaze360.pth.tar', map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        # dictionaries to track number of people
        self.person_dict = {i:[] for i in range(MAX_PEOPLE)}
        self.empty_frames = [i for i in range(MAX_PEOPLE)]

        # camera capture to read images
        self.cap = cv2.VideoCapture(0)

        # mediapipe imports
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    def spherical2cartesial(x):    
        output = torch.zeros(x.size(0),3)
        output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
        output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
        output[:,1] = torch.sin(x[:,1])
        return output

    def get_eye_bbox(detection, height, width):
        '''
        returns the position of eye gaze(x, y) and the bounding box around the detected face(x0, x1, y0, y1)
        '''
        xmin = (detection.location_data.relative_bounding_box.xmin)
        ymin = (detection.location_data.relative_bounding_box.ymin)
        widthF =(detection.location_data.relative_bounding_box.width)
        heightF = (detection.location_data.relative_bounding_box.height)
        
        x0 = math.floor(max(0, xmin - widthF*0.5)*width)
        x1 = math.floor(min(width, x0 + (widthF+widthF)*width))
        y0 = math.floor(max(0, ymin - heightF*0.5)*height)
        y1 = math.floor(min(height, y0+(heightF+ heightF)*height))

        return ([(x0+x1)/2.0, (0.65*y0+0.35*y1)], [x0, x1, y0, y1])


    def process_image(self, image: np.array) -> np.array:
        '''
        Take the image, compute the gaze along with the bounding boxand return the image with gaze drawn back
        '''
        h, w, c  = image.shape
        max_people = max(self.person_dict.keys())
        results = self.face_detection.process(image)
        if results.detections:
            
            for i in range(len(results.detection), max_people):
                self.empty_frames[i] += 1

            for i, detection in enumerate(results.detections):
                if i > max_people:
                    # ignore the people other than the max number of people
                    continue

                self.empty_frames[i] = 0
                # get the eye pos and bounding box from the detection
                eyePos, boundBox = self.get_eye_pos(detection, h, w)
                # crop the faceImage and store it in the person_dict 
                faceImage = self.transformation(image[boundBox[2]:boundBox[3], boundBox[0]:boundBox[1]])
                self.person_dict[i].append((faceImage, eyePos))
        else:
            for i in range(max_people):
                self.empty_frames[i] += 1

        # flush the array where faces are not being detected
        for i in range(max_people):
            if self.empty_frames[i] >  self.FLUSH_AFTER_N_FRAMES:
                self.person_dict[i] = []  
        
        process_faces = []
        
        for i, face_image_batch in enumerate(self.person_dict.values()):
            if len(face_image_batch) == 7:
                process_faces.append(torch.stack(face_image_batch).reshape(21, 224, 224))
                self.person_dict[i].pop(0)
        
        if len(process_faces) == 0:
            return image
        
        process_faces = torch.stack(process_faces).to(self.device)
        gazeVector, _ = self.model(process_faces)
        gazeVector = gazeVector.detach().cpu() 
        return image


    def run(self):
        while True:
            start_time = time.time()
            
            success, image = self.cap.read()
        
            if not success:
                print("Image Not Read!")
                continue

            image = self.process_image(image)        
            
            cv2.imshow("X", image)
            if cv2.waitKey(1) & 0xFF == 27:
                self.cap.release()
                cv2.destroyAllWindows()
                break
            
            end_time = time.time()