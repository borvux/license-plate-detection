import csv
import numpy as np
from scipy.interpolate import interp1d

class DataStabilzer:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    # stablize the data of the results.csv
    def stabilize_data(self, input_data):
        # get the data from results.csv
        frame_nums = np.array([int(row['frame_number']) for row in input_data])
        vehicle_ids = np.array([int(float(row['vehicle_id'])) for row in input_data])
        vehicle_bboxes = np.array([list(map(float, row['vehicle_bonding_box'][1:-1].split())) for row in input_data])
        license_bboxes = np.array([list(map(float, row['license_plate_bonding_box'][1:-1].split())) for row in input_data])

        # get the unique ids from the input data
        interpolated_output = []
        unique_vehicle_ids = np.unique(vehicle_ids)
        for vehicle_id in unique_vehicle_ids:
            frame_nums_ = [p['frame_number'] for p in input_data if int(float(p['vehicle_id'])) == int(float(vehicle_id))]
            print(frame_nums_, vehicle_id)

            # filter out the current data 
            vehicle_mask = vehicle_ids == vehicle_id
            vehicle_frame_nums = frame_nums[vehicle_mask]
            vehicle_bboxes_interpolated = []
            license_bboxes_interpolated = []

            start_frame_num = vehicle_frame_nums[0]
            end_frame_num = vehicle_frame_nums[-1]

            # start to fill in the missing data 
            for i in range(len(vehicle_bboxes[vehicle_mask])):
                frame_num = vehicle_frame_nums[i]
                vehicle_bbox = vehicle_bboxes[vehicle_mask][i]
                license_bbox = license_bboxes[vehicle_mask][i]
                
                # ff it’s not the first bounding box, the previous frame number, vehicle and license plate bounding box are extracted
                if i > 0:
                    prev_frame_num = vehicle_frame_nums[i-1]
                    prev_vehicle_bbox = vehicle_bboxes_interpolated[-1]
                    prev_license_bbox = license_bboxes_interpolated[-1]

                    # if there are missing data and gaps, fill those in 
                    if frame_num - prev_frame_num > 1:
                        gap_in_frames = frame_num - prev_frame_num
                        x = np.array([prev_frame_num, frame_num])
                        x_new = np.linspace(prev_frame_num, frame_num, num=gap_in_frames, endpoint=False)
                        interp_func = interp1d(x, np.vstack((prev_vehicle_bbox, vehicle_bbox)), axis=0, kind='linear')
                        interpolated_vehicle_bboxes = interp_func(x_new)
                        interp_func = interp1d(x, np.vstack((prev_license_bbox, license_bbox)), axis=0, kind='linear')
                        interpolated_license_bboxes = interp_func(x_new)
                        vehicle_bboxes_interpolated.extend(interpolated_vehicle_bboxes[1:])
                        license_bboxes_interpolated.extend(interpolated_license_bboxes[1:])
                        
                # add the missing data for both the vehicle and license plates 
                vehicle_bboxes_interpolated.append(vehicle_bbox)
                license_bboxes_interpolated.append(license_bbox)

            # create new rows for the missing datas 
            for i in range(len(vehicle_bboxes_interpolated)):
                frame_num = start_frame_num + i
                row = {}
                row['frame_number'] = str(frame_num)
                row['vehicle_id'] = str(vehicle_id)
                row['vehicle_bonding_box'] = ' '.join(map(str, vehicle_bboxes_interpolated[i]))
                row['license_plate_bonding_box'] = ' '.join(map(str, license_bboxes_interpolated[i]))

                # if the frame number is not the same as original, the license plate bounding box score, character, and character score are set to ‘0’
                if str(frame_num) not in frame_nums_:
                    row['license_plate_bonding_box_score'] = '0'
                    row['license_plate_character'] = '0'
                    row['license_plate_character_score'] = '0'
                    
                # else, they are extracted from the original row
                else:
                    original_row = [p for p in input_data if int(p['frame_number']) == frame_num and int(float(p['vehicle_id'])) == int(float(vehicle_id))][0]
                    row['license_plate_bonding_box_score'] = original_row['license_plate_bonding_box_score'] if 'license_plate_bonding_box_score' in original_row else '0'
                    row['license_plate_character'] = original_row['license_plate_character'] if 'license_plate_character' in original_row else '0'
                    row['license_plate_character_score'] = original_row['license_plate_character_score'] if 'license_plate_character_score' in original_row else '0'

                interpolated_output.append(row)

        return interpolated_output

    # process method
    def process(self):
        # read the input file
        with open(self.input_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            input_data = list(csv_reader)

        # stablize the data
        interpolated_output = self.stabilize_data(input_data)

        # write to the new csv file that is stablize
        csv_header = ['frame_number', 'vehicle_id', 'vehicle_bonding_box', 'license_plate_bonding_box', 'license_plate_bonding_box_score', 'license_plate_character', 'license_plate_character_score']
        with open(self.output_file, 'w', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=csv_header)
            csv_writer.writeheader()
            csv_writer.writerows(interpolated_output)

# stabalize the data
interpolator = DataStabilzer(
    'output/results.csv',           # path of the original csv
    'output/results_stabilize.csv') # output path of new stabalize csv
interpolator.process()