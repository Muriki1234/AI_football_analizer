import math
seg_durations = [5.0, 5.0, 5.0]
lines = [
    "#EXTM3U",
    "#EXT-X-VERSION:3",
    f"#EXT-X-TARGETDURATION:{math.ceil(max(seg_durations))}",
    "#EXT-X-MEDIA-SEQUENCE:0",
    "#EXT-X-PLAYLIST-TYPE:VOD"
]
for i in range(len(seg_durations)):
    lines.append("#EXT-X-DISCONTINUITY")
    lines.append(f"#EXTINF:{seg_durations[i]:.3f},")
    lines.append(f"chunk_{i:03d}.ts")
lines.append("#EXT-X-ENDLIST")
print("\n".join(lines))
