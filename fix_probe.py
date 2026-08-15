import subprocess
import math

def _probe_total_frames(path: str, fps: float) -> int:
    try:
        # Try to get nb_frames from video stream
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if out and out.isdigit():
            return int(out)
    except Exception:
        pass

    try:
        # Fallback to duration * fps
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if out and out != 'N/A':
            return math.ceil(float(out) * fps)
    except Exception:
        pass

    return 0
print("Working")
