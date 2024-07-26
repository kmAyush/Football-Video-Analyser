from utils import measure_distance, get_foot_position
import cv2

class MotionEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == 'ball' or object == 'referees':
                continue
            num_frames = len(object_tracks)
            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames-1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = object_tracks[frame_num][track_id]['position_transformed']
                    end_pos = object_tracks[last_frame][track_id]['position_transformed']

                    if start_pos is None or end_pos is None:
                        continue

                    distance_covered = measure_distance(start_pos, end_pos)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    speed_ms = distance_covered/time_elapsed
                    speed_kmph = speed_ms*3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue

                        tracks[object][frame_num_batch][track_id]['speed'] = speed_kmph
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == 'ball' or object == 'referees':
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)

                        if speed is None or distance is None:
                            continue
                        
                        bbox = track_info["bounding_box"]
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1]+=40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (117,30,54), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,20,20), 2)
            output_frames.append(frame)

        return output_frames    

