'''
pip install numpy
pip install pandas
pip install ultralytics
pip install opencv-python
pip install easyocr
pip install matplotlib

for sort libary to work:
    pip install filterpy
    pip install lap, if only python 3.9 or lower
    pip install pytest

pip install ultralytics pandas opencv-python numpy scipy easyocr filterpy

if windows:
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else on mac:
    pip3 install torch torchvision torchaudio
    
***** TO RUN THIS PROGRAM *****
    - Add a video with license plate you want to detect in the output folder
    - Run the programs in this order:
        1. main.py - will create results.csv
        2. stabilizer.py - will create results_stabilize.csv
        3. output.py - will create output.mp4
'''
# import libaries
from ultralytics import YOLO
import cv2
from sort.sort import *
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import string
import easyocr

# check if we can use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VehicleTracker:
    def __init__(self, coco_model_path, license_plate_detector_path, video_path):
        # Initialize the pre-trained COCO model, OCR, the train license plate detection model, cv2 and sort 
        self.coco_model = YOLO(coco_model_path).to(device)
        self.license_plate_detector = YOLO(license_plate_detector_path).to(device)
        self.capture = cv2.VideoCapture(video_path)
        self.motion_tracker = Sort()
        self.reader = easyocr.Reader(["en"], gpu=True, quantize=False)
        '''
        2: car
        3: motorcycle
        5: bus
        7: truck
        '''
        self.class_ids = [2, 3, 5, 7]
        # result dictionary to write to result.csv
        self.results = {}

    # write to csv function
    def write_csv(self, output_path):
        with open(output_path, 'w') as f:
            f.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'vehicle_id', 'vehicle_bonding_box',
                                                    'license_plate_bonding_box', 'license_plate_bonding_box_score', 
                                                    'license_plate_character', 'license_plate_character_score'))
            for frame_nmr in self.results.keys():
                for car_id in self.results[frame_nmr].keys():
                    print(self.results[frame_nmr][car_id])
                    if 'vehicle' in self.results[frame_nmr][car_id].keys() and \
                       'license_plate' in self.results[frame_nmr][car_id].keys() and \
                       'character' in self.results[frame_nmr][car_id]['license_plate'].keys():
                        f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr, car_id, '[{} {} {} {}]'.format(self.results[frame_nmr][car_id]['vehicle']['bonding_box'][0],
                                                                                                          self.results[frame_nmr][car_id]['vehicle']['bonding_box'][1],
                                                                                                          self.results[frame_nmr][car_id]['vehicle']['bonding_box'][2],
                                                                                                          self.results[frame_nmr][car_id]['vehicle']['bonding_box'][3]),
                                                                                   '[{} {} {} {}]'.format(self.results[frame_nmr][car_id]['license_plate']['bonding_box'][0],
                                                                                                          self.results[frame_nmr][car_id]['license_plate']['bonding_box'][1],
                                                                                                          self.results[frame_nmr][car_id]['license_plate']['bonding_box'][2],
                                                                                                          self.results[frame_nmr][car_id]['license_plate']['bonding_box'][3]),
                                                                                                          self.results[frame_nmr][car_id]['license_plate']['bonding_box_score'],
                                                                                                          self.results[frame_nmr][car_id]['license_plate']['character'],
                                                                                                          self.results[frame_nmr][car_id]['license_plate']['character_score']))
            f.close()

    # reading the license plate
    def read_license_plate(self, license_plate_crop):
        # format method to see if text is less than equal to 8 with either string or int
        def format(text):
            return len(text) <= 8 and all(char in string.ascii_uppercase or char.isdigit() for char in text)
        
        # read the license plate using easyocr
        characters = self.reader.readtext(license_plate_crop)
        formatted_characters = [(text.upper().replace(' ', ''), score) for bbox, text, score in characters if format(text.upper().replace(' ', ''))]
        # return the formmated characters 
        return next(iter(formatted_characters), (None, None))
    
    # get vehicle method 
    def get_vehicle(self, license_plate, vehicle_track_ids):
        # assigns the bounding box of license plate
        x1, y1, x2, y2, score, class_id = license_plate
        # check if license plate boxes are in the vehcile 
        car = next((vehicle for vehicle in vehicle_track_ids if x1 > vehicle[0] and y1 > vehicle[1] and x2 < vehicle[2] and y2 < vehicle[3]), None)
        return car if car is not None else (-1, -1, -1, -1, -1) # return -1 if no vehicle

    # track vechicles method 
    def track_vehicles(self):
        frame_number = -1
        ret = True

        # go through each frame of the video
        while ret:
            frame_number += 1
            ret, frame = self.capture.read()

            #if ret and frame_number < 120:
            if ret:
                self.results[frame_number] = {}
                detected = []
                object_detections = self.coco_model(frame)[0]

                # get the bounding boxes of the vehicle, confidence score, and class id 
                for object_detection in object_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = object_detection

                    # add them to the detected list when the class id matched with class id list above
                    if int(class_id) in self.class_ids:
                        detected.append([x1, y1, x2, y2, score])
                
                # track the vehicles, and track their license plates 
                track_ids = self.motion_tracker.update(np.asarray(detected))
                license_plates_detections = self.license_plate_detector(frame)[0]

                # get the bounding boxes of the license plates 
                for license_plate in license_plates_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    # make sure the vehicle boxes and license plate boxes matches
                    xv1, yv1, xv2, yv2, vehicle_id = self.get_vehicle(license_plate, track_ids)

                    if vehicle_id != 1:
                        # process the license plate and read the license plate number
                        license_plate_crop = frame[int(y1):int(y2),int(x1):int(x2),:]
                        license_plate_BW = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        # _, license_plate_threshold = cv2.threshold(license_plate_BW, 64, 255, cv2.THRESH_BINARY_INV)
                        # characters, characters_score = self.read_license_plate(license_plate_threshold)
                        characters, characters_score = self.read_license_plate(license_plate_BW)
                        
                        if characters is not None:
                            # put the results into the dictionary
                            self.results[frame_number][vehicle_id] = {'vehicle': {'bonding_box': [xv1, yv1, xv2, yv2]},
                                                                        'license_plate': {'bonding_box': [x1, y1, x2, y2],
                                                                                        'character' : characters,
                                                                                        'bonding_box_score' : score,
                                                                                        'character_score' : characters_score}}
        return self.results
    
if __name__ == "__main__":
    # yolov8n.pt : pre-trained COCO model
    # LPD_best_train.pt : our trained license plate detection model
    tracker = VehicleTracker(
        'models/yolov8n.pt',        # common objects in context model
        'models/LPD_best_train.pt', # license plate 
        'output/IMG_8033.mp4')      # path of the video to detect
    
    # get the results 
    results = tracker.track_vehicles()
    tracker.write_csv('output/results.csv')