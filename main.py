#!/usr/bin/env python3 
from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video('input/football_sample.mp4')

    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'output/output.avi')

if __name__ == '__main__':
    main()