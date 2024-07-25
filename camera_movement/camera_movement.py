from utils import measure_distance, measure_xy_distance
import pickle
import cv2
import numpy as np
import os

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1205] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )
    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']

                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub = False, stub_path = None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        
        camera_motion = [[0,0]]*len(frames)

        first_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        first_features = cv2.goodFeaturesToTrack(first_frame, **self.features)

        for frame_num in range(1, len(frames)):
            new_frame = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(first_frame, new_frame, first_features, None, **self.lk_params)
            
            max_distance = 0
            camera_move_x, camera_move_y = 0, 0
            for i, (new, old) in enumerate(zip(new_features, first_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_move_x, camera_move_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_motion[frame_num] = [camera_move_x, camera_move_y]
                first_features = cv2.goodFeaturesToTrack(new_frame, **self.features)

            first_frame = new_frame.copy()
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_motion, f)
    
        return camera_motion

    def draw_camera_movement(self, frames, camera_move_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            x_move, y_move = camera_move_per_frame[frame_num]

            frame = cv2.putText(frame, f"Camera Movement X : {x_move:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y : {y_move:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

            output_frames.append(frame)
        
        return output_frames