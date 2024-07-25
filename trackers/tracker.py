from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
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
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bounding_box',[]) for x in ball_positions]
        df_bp = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        df_bp = df_bp.interpolate()
        df_bp = df_bp.bfill()

        ball_positions = [{1: {"bounding_box":x}} for x in df_bp.to_numpy().tolist()]
        
        return ball_positions
    
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

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame,(int(x1_rect), int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (0,0,0),
                thickness = 2
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2 )

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
                color = player.get("team_color", (200, 99, 56))
                frame = self.draw_ellipse(frame, player["bounding_box"], color, track_id)
            
            # Draw Referee
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bounding_box"],(0,255,255), track_id)
            
            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bounding_box"], (0,255,0))

            output_video_frames.append(frame)
        
        return output_video_frames