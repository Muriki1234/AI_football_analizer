import os
import sys
import argparse
import subprocess
import pickle
import tempfile
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ================= Production Config =================
SKIP_STEP = 3        # Skip every N frames (3-5 is good for football)
TARGET_HEIGHT = 480   # Resize height for tracking to save memory/time
# =====================================================

def interpolate_bboxes(sparse_data, total_frames, scale_factor=1.0):
    """Interpolate missing frames and scale back to original dimensions."""
    df = pd.DataFrame(index=range(total_frames), columns=['x', 'y', 'w', 'h'])
    for idx, bbox in sparse_data.items():
        if idx < total_frames:
            # bbox is [x, y, w, h] from tracker
            df.loc[idx] = [b * scale_factor for b in bbox]

    df = df.astype(float).interpolate(method='linear', limit_direction='both').bfill().ffill()
    
    result = {}
    for idx, row in df.iterrows():
        result[idx] = [int(row['x']), int(row['y']), int(row['w']), int(row['h'])]
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--bbox", required=True) # x,y,w,h (original scale)
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--output_pkl", required=True)
    args = parser.parse_args()

    # 1. Get original video info
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Calculate scale factor
    resize_factor = TARGET_HEIGHT / orig_h if orig_h > TARGET_HEIGHT else 1.0
    new_w = int(orig_w * resize_factor)
    new_h = int(orig_h * resize_factor)
    
    # Scale initial bbox for tracker
    ox, oy, ow, oh = map(float, args.bbox.split(','))
    scaled_bbox_str = f"{int(ox*resize_factor)},{int(oy*resize_factor)},{int(ow*resize_factor)},{int(oh*resize_factor)}"

    # 2. Create temporary workspace
    tmp_dir = tempfile.mkdtemp(prefix="samurai_work_")
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    prompt_txt_path = os.path.join(tmp_dir, "prompt.txt")
    dummy_mp4_path = os.path.join(tmp_dir, "output.mp4")
    bbox_out_path = dummy_mp4_path.replace(".mp4", "_bboxes.txt")

    try:
        # 3. FFmpeg Extraction (Resized & Skipped)
        # We extract frames 0, 3, 6... into the frames_dir
        print(f"🔄 Extracting frames (Skip={SKIP_STEP}, Resize={new_w}x{new_h})...")
        ffmpeg_cmd = [
            "ffmpeg", "-i", args.video, "-y",
            "-vf", f"select='not(mod(n\\,{SKIP_STEP}))',scale={new_w}:{new_h}",
            "-vsync", "0",
            "-q:v", "2",
            f"{frames_dir}/%05d.jpg"
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

        # 4. Run SAMURAI demo.py
        samurai_dir = os.path.dirname(os.path.abspath(__file__))
        demo_script = os.path.join(samurai_dir, "scripts", "demo.py")
        
        with open(prompt_txt_path, "w") as f:
            f.write(f"{scaled_bbox_str}\n")

        # Command to run on the extracted frame directory
        cmd = [
            sys.executable, demo_script,
            "--video_path", frames_dir,
            "--txt_path", prompt_txt_path,
            "--video_output_path", dummy_mp4_path,
            # We don't need the output video, just the bboxes
        ]

        print(f"🚀 Running SAMURAI tracking on {len(os.listdir(frames_dir))} frames...")
        subprocess.run(cmd, cwd=samurai_dir, capture_output=True, check=True)

        # 5. Map results back and Interpolate
        if not os.path.exists(bbox_out_path):
            raise FileNotFoundError(f"SAMURAI did not produce {bbox_out_path}")

        sparse_results = {}
        with open(bbox_out_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    # demo.py frame_idx is relative to the directory (0, 1, 2...)
                    # mapping back to global: local_idx * SKIP_STEP
                    local_idx = int(parts[0])
                    global_idx = local_idx * SKIP_STEP
                    x, y, w, h = map(float, parts[2:6])
                    if w > 0 and h > 0:
                        sparse_results[global_idx] = (x, y, w, h)

        print(f"📈 Interpolating {len(sparse_results)} points to {total_frames} frames...")
        full_bboxes = interpolate_bboxes(sparse_results, total_frames, scale_factor=1/resize_factor)

        # 6. Save to final pkl
        os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)
        with open(args.output_pkl, "wb") as f:
            pickle.dump({"bboxes": full_bboxes}, f)
            
        print(f"✅ Success! Saved {len(full_bboxes)} frames to {args.output_pkl}")

    finally:
        # Cleanup
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
