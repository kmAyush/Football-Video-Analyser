from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
sys.path.append('../')
from utils import get_bbox_width, get_center_bbox


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size = 20
        results = []
        for i in range(0,len(frames), batch_size):
            result = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            results +=  result
        return results


    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        # tracks = {
        #     Player : [
        #         {// Frame 1
        #             0:{"bbox":[0,0,0,1]}, 1:{"bbox":[4,4,0,1]}
        #         },{ // Frame 2
        #             3:{"bbox":[0,0,0,1]}, 4:{"bbox":[4,4,0,1]}
        #         }
        #     ]
        # }
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        for frame_num, detection in enumerate(detections):
            class_name  = detection.names # 0 - person, 1 - goal, 2 - referee
            class_name_inverse = {value:key for key, value in class_name.items()}

            # Convert to supervised detection format
            detection_supervised = sv.Detections.from_ultralytics(detection)

            # Consider Goalkeeper as player 
            for obj, class_id in enumerate(detection_supervised.class_id):
                if class_name[class_id] == "goalkeeper":
                   detection_supervised.class_id[obj] = class_name_inverse["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervised)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detect in detection_with_tracks:
                bounding_box = frame_detect[0].tolist()
                class_id = frame_detect[3]
                track_id = frame_detect[4]

                if class_id == class_name_inverse["player"]:
                    tracks["players"][frame_num][track_id] = {"bounding_box":bounding_box}

                if class_id == class_name_inverse["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bounding_box":bounding_box}    

            for frame_detect in detection_supervised:
                bbox = frame_detect[0].tolist()
                class_id = frame_detect[3]

                if class_id == class_name_inverse["ball"]:
                    tracks["ball"][frame_num][1] = {"bounding_box":bbox}
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks
            
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        x_center, _ = get_center_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, center = (x_center, y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = 45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType= cv2.LINE_4
        )
        return frame



    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bounding_box"],(0,0,255), track_id)
            
            output_video_frames.append(frame)
        
        return output_video_frames