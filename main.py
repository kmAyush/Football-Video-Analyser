#!/usr/bin/env python3 
from utils import read_video, save_video
from trackers import Tracker
from assign_team import AssignTeam
import cv2

def main():
    video_frames = read_video('input/football_sample.mp4')

    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    # Interpolating ball

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Generating Person Image
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player["bounding_box"]
    #     frame = video_frames[0]

    #     croppedImage = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    #     cv2.imwrite(f'output/cropped.jpg',croppedImage)
    #     break

    team_assigner = AssignTeam()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team =  team_assigner.get_player_team(video_frames[frame_num], track['bounding_box'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'output/output.avi')

if __name__ == '__main__':
    main()