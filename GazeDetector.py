import math, time, cv2, torch
import socket, struct, pickle
import mediapipe as mp
import numpy as np
from PIL import Image
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
        self.model = torch.nn.DataParallel(model)
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
    

    def spherical2cartesial(self, x):    
        output = torch.zeros(x.size(0),3)
        output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
        output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
        output[:,1] = torch.sin(x[:,1])
        return output
    

    def draw_gaze(self, pos,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
        
        """
        a= x_min
        b = y_min
        c =width
        d= height
        Draw gaze angle on given image with a given eye positions."""

        image_out = image_in
        (h, w) = image_in.shape[:2]
        length = w/2
        
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(math.radians(pitchyaw[0])) * np.cos(math.radians(pitchyaw[1]))
        dy = -length * np.sin(math.radians(pitchyaw[1]))

        cv2.arrowedLine(image_out, (np.round(pos).astype(np.int32)),
                    (np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.18)
        return image_out    


    def get_eye_bbox(self, detection, height, width):
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

        return (((x0+x1)/2.0, (0.65*y0+0.35*y1)), [x0, x1, y0, y1])


    def process_image(self, image: np.array) -> np.array:
        '''
        Take the image, compute the gaze along with the bounding boxand return the image with gaze drawn back
        '''
        h, w, c  = image.shape
        max_people = max(self.person_dict.keys())
        results = self.face_detection.process(image)
        if results.detections:
            
            for i in range(len(results.detections), max_people):
                self.empty_frames[i] += 1

            for i, detection in enumerate(results.detections):
                if i > max_people:
                    # ignore the people other than the max number of people
                    continue

                self.empty_frames[i] = 0
                # get the eye pos and bounding box from the detection
                eyePos, boundBox = self.get_eye_bbox(detection, h, w)
                # crop the faceImage and store it in the person_dict 
                faceImage = self.transformation(Image.fromarray(image[boundBox[2]:boundBox[3], boundBox[0]:boundBox[1]]))
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
                face_image_batch = [face[0] for face in face_image_batch]
                process_faces.append(torch.stack(face_image_batch).reshape(21, 224, 224))
                self.person_dict[i].pop(0)
        
        if len(process_faces) == 0:
            return image
        
        process_faces = torch.stack(process_faces).to(self.device)
        gazeVectors, _ = self.model(process_faces)
        gazeVectors = gazeVectors.detach().cpu()
        gazeVectors = self.spherical2cartesial(gazeVectors)
        
        for i, gaze_vector in enumerate(gazeVectors):
            eyePos = self.person_dict[i][3][1]
            tdx = int(eyePos[0])
            tdy = int(eyePos[1])
            self.draw_gaze((tdx, tdy), image, [math.degrees(gaze_vector[0]), math.degrees(gaze_vector[1])])
        
        print("_________________________________________________")
        print(f"Detected {process_faces.shape[0]} Faces")
        print(f"gaze_vectors: ", gazeVectors)
        return image

    
    def run_server(self):

        # release the camera object
        self.cap.release()
  

        server_ip = '127.0.0.1'  # Change this to your server IP
        server_port = 9999

        # Create socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((server_ip, server_port))
        server_socket.listen(5)

        # Accept connection from client
        client_socket, addr = server_socket.accept()
        print("Connection from: ", addr)

        # Start receiving video stream
        data = b""
        payload_size = struct.calcsize("L")

        while True:
            while len(data) < payload_size:
                data += client_socket.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Deserialize frame data
            frame = pickle.loads(frame_data)

            frame = self.process_image(frame)
            
            # Display received frame
            cv2.imshow('Server', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.cap.release()
                cv2.destroyAllWindows()
                break

            # Send acknowledgment to client
            client_socket.send(b"OK")

        # Close sockets
        client_socket.close()
        server_socket.close()



    def run_local(self):
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
            fps = 1/(end_time - start_time)
            print("FPS: ", fps)