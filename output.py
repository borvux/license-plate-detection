import ast
import cv2
import numpy as np
import pandas as pd

class VideoProcessor:
    def __init__(self, video_path, results_path, output_path):
        self.video_path = video_path
        self.results = pd.read_csv(results_path)
        self.cap = cv2.VideoCapture(self.video_path)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (self.width, self.height))
        self.license_plate = {}
        self.frame_number = -1

    @staticmethod # draw bounding boxes method
    def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=1, line_length_x=10, line_length_y=10):
        x1, y1 = top_left
        x2, y2 = bottom_right

        # draw the each lines corresponding to the boxes 
        cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
        cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
        cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
        cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
        cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

        return img

    # process the video method
    def process_video(self):
        # loops over each unique vehicle ids in the results
        for car_id in np.unique(self.results['vehicle_id']):
            
            # finds the maximum license plate character score and stores the corresponding license plate character and a cropped image of the license plate
            max_score = np.amax(self.results[self.results['vehicle_id'] == car_id]['license_plate_character_score'])
            self.license_plate[car_id] = {'license_crop': None, 'license_plate_character': self.results[(self.results['vehicle_id'] == car_id) & (self.results['license_plate_character_score'] == max_score)]['license_plate_character'].iloc[0]}
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.results[(self.results['vehicle_id'] == car_id) & (self.results['license_plate_character_score'] == max_score)]['frame_number'].iloc[0])
            ret, frame = self.cap.read()

            x1, y1, x2, y2 = ast.literal_eval(self.results[(self.results['vehicle_id'] == car_id) & (self.results['license_plate_character_score'] == max_score)]['license_plate_bonding_box'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

            # get the crop license plate
            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 50 / (y2 - y1)), 50))
            self.license_plate[car_id]['license_crop'] = license_crop

        # resets the position of the video frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # go through all the frames again
        ret = True
        while ret:
            ret, frame = self.cap.read()
            self.frame_number += 1
            
            # if there is a frame, get the rows of results corresponding to the current frame number
            if ret:
                df_ = self.results[self.results['frame_number'] == self.frame_number]
                
                # in reach row of the results
                for row_indx in range(len(df_)):
                    # get the vehicle boudning boxes
                    car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['vehicle_bonding_box'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    # and draw it 
                    self.draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 10, line_length_x=50, line_length_y=50)

                    # get the license plate bounding boxes 
                    x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bonding_box'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

                    # get the image of the license plate
                    license_crop = self.license_plate[df_.iloc[row_indx]['vehicle_id']]['license_crop']
                    
                    # get the dimensions of the licnese plate image
                    H, W, _ = license_crop.shape

                    try:
                        # overlay the license plate above the vehicle
                        frame[int(car_y1) - H - 50:int(car_y1) - 50, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
                        frame[int(car_y1) - H - 50:int(car_y1) - H - 50, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                        # overlay the license plate character below the license plate image
                        (text_width, text_height), _ = cv2.getTextSize(self.license_plate[df_.iloc[row_indx]['vehicle_id']]['license_plate_character'],cv2.FONT_HERSHEY_SIMPLEX,1,1)
                        cv2.putText(frame,self.license_plate[df_.iloc[row_indx]['vehicle_id']]['license_plate_character'],(int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H + (text_height / 2))),cv2.FONT_HERSHEY_PLAIN,2,(255, 255, 255),2)

                    except:
                        pass
                
                # writes the new frames
                self.out.write(frame)

        self.out.release()
        self.cap.release()

# output = VideoProcessor('output/IMG_8033.mp4', 'output/results.csv', 'output/output_NS.mp4')
# process a new output video           
output = VideoProcessor(
    'output/IMG_8033.mp4',          # path of the input video
    'output/results_stabilize.csv', # path of the stabalize data csv
    'output/output.mp4')            # path of the output video
output.process_video()