#!/usr/bin/env python3 
from utils import read_video, save_video
from trackers import Tracker
from assign_team import AssignTeam
from assign_ball import AssignPlayerBall
from camera_movement import CameraMovementEstimator
from view_transformer import ViewTransformer
from motion_estimator import MotionEstimator
import numpy as np
import cv2

def main():
    video_frames = read_video('input/football_sample.mp4')

    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Get Object Positions
    tracker.add_position_to_tracks(tracks)

    # Estimating Camera Movement
    camera_estimate = CameraMovementEstimator(video_frames[0])
    camera_move_per_frame = camera_estimate.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')

    # Adjust positions to track
    camera_estimate.add_adjust_positions_to_tracks(tracks, camera_move_per_frame)

    # View Transformer
    v_transformer = ViewTransformer()
    v_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolating ball
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Motion Estimator
    motion_estimator = MotionEstimator()
    motion_estimator.add_speed_distance_to_tracks(tracks)

    team_assigner = AssignTeam()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # Assign teams
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team =  team_assigner.get_player_team(video_frames[frame_num], track['bounding_box'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign ball acquisition
    player_assigner = AssignPlayerBall()
    ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bounding_box']
        player_assigned  = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # Assign Control
        if player_assigned != -1:
            tracks['players'][frame_num][player_assigned]['has_ball'] = True
            ball_control.append(tracks['players'][frame_num][player_assigned]['team'])
        else:
            ball_control.append(ball_control[-1])
    
    ball_control = np.array(ball_control)
            
    # Draw Annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, ball_control)
    output_video_frames = camera_estimate.draw_camera_movement(output_video_frames, camera_move_per_frame)
    motion_estimator.draw_speed_distance(output_video_frames, tracks)

    save_video(output_video_frames, 'output/output.avi')

if __name__ == '__main__':
    main()