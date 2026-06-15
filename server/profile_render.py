import time
import cv2
import pickle
import numpy as np
import sys
import os

from server.pipeline.analysis_core import Tracker, TeamAssigner, SoccerPitchConfiguration

def profile():
    print("Loading tracks...")
    # Need a session dir
    base_dir = "/workspace/outputs/37f9aeaf-e7c5-4997-b9b4-a500013de7ce"
    if not os.path.exists(base_dir):
        print(f"Path not found: {base_dir}")
        return
        
    with open(os.path.join(base_dir, "tracks.pkl"), "rb") as f:
        data = pickle.load(f)
        
    tracks = data["tracks"]
    tracked_bboxes = data["tracked_bboxes"]
    team_control = data["team_control"]
    
    tracker_obj = Tracker(None)
    team_assigner = TeamAssigner()
    team_assigner.team_colors = {
        1: np.array([255, 0, 0]),
        2: np.array([0, 0, 255]),
    }
    config = SoccerPitchConfiguration()
    
    print("Reading video...")
    cap = cv2.VideoCapture(os.path.join(base_dir, "video.mp4")) # wait, what is video_path?
    # I don't know the video path. I'll just create a dummy frame!
    frame = np.zeros((720, 1152, 3), dtype=np.uint8)
    
    from server.pipeline.tasks import _render_single_frame_worker_full
    
    hex_t1, hex_t2 = "#FF0000", "#0000FF"
    
    print("Profiling 100 frames...")
    start = time.perf_counter()
    for i in range(100):
        rargs = (i, frame, tracks, tracked_bboxes, team_control, team_assigner, tracker_obj, config, hex_t1, hex_t2)
        _render_single_frame_worker_full(rargs)
    end = time.perf_counter()
    
    print(f"100 frames took {end-start:.4f} seconds.")
    print(f"That's {(end-start)/100:.4f} s/frame.")
    
if __name__ == "__main__":
    profile()
