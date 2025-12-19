import cv2
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# User settings
# ----------------------------------------------------------
CSV_PATH = "data.csv"
VIDEO_PATH = "vid.mp4"
OUTPUT_VIDEO = "telemetry_video_g.mp4"

CIRCLE_RADIUS = 120
G_MAX_DEFAULT = 1.5

# Visual tuning
SMOOTH_ALPHA = 0.12
TRAIL_LENGTH = 20
AUTO_G_MAX = True
G_MAX_MIN = 1.2
G_MAX_MAX = 2.5
G_SCALE_RATE = 0.02

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
LIGHT_GRAY = (180, 180, 180)

# ----------------------------------------------------------
# Load telemetry
# ----------------------------------------------------------
df = pd.read_csv(CSV_PATH)

telemetry_time = df["Time"].values.astype(np.float64) / 1000.0
imu_long = df["imu.long"].values.astype(np.float32)
imu_lat  = df["imu.lat"].values.astype(np.float32)

# Remove NaNs
valid = ~(np.isnan(imu_long) | np.isnan(imu_lat))
telemetry_time = telemetry_time[valid]
imu_long = imu_long[valid]
imu_lat  = imu_lat[valid]

# Convert m/s² → g ONCE
imu_long /= 9.8
imu_lat  /= 9.8

# ----------------------------------------------------------
# Load video
# ----------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

video_duration = frame_cnt / fps

telemetry_end = telemetry_time[-1]
video_start = telemetry_end - video_duration

print(f"Video duration: {video_duration:.3f}s")
print(f"Video start: {video_start:.3f}")
print(f"Video end:   {telemetry_end:.3f}")

# ----------------------------------------------------------
# G-circle placement
# ----------------------------------------------------------
CIRCLE_CENTER = (width - 160, height - 160)

# ----------------------------------------------------------
# Video writer
# ----------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

VIDEO_TIME_OFFSET = -4

print('Vid time offset: {VIDEO_TIME_OFFSET}')

# ----------------------------------------------------------
# State variables
# ----------------------------------------------------------
smooth_lat = 0.0
smooth_long = 0.0
first_sample = True
current_g_max = G_MAX_DEFAULT
trail = []

# ----------------------------------------------------------
# Fast G-circle renderer (no full-frame blending)
# ----------------------------------------------------------
def draw_g_circle(frame, lat_g, long_g, center, radius, g_max, trail):
    cx, cy = center

    # Outer ring
    cv2.circle(frame, center, radius, LIGHT_GRAY, 2)

    # Axes
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), GRAY, 1)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), GRAY, 1)

    # Tick rings
    for g in (0.5, 1.0):
        if g < g_max:
            r = int(radius * g / g_max)
            cv2.circle(frame, center, r, (80, 80, 80), 1)

    lat = max(-g_max, min(lat_g, g_max))
    lon = max(-g_max, min(long_g, g_max))

    px = cx + (lat / g_max) * radius
    py = cy - (lon / g_max) * radius

    g_mag = (lat * lat + lon * lon) ** 0.5

    if g_mag < 0.6:
        color = (0, 180, 0)
    elif g_mag < 1.0:
        color = (0, 200, 220)
    else:
        color = (0, 0, 220)

    # Trail (fast, no blending)
    trail.append((px, py))
    if len(trail) > TRAIL_LENGTH:
        trail.pop(0)

    for i, (tx, ty) in enumerate(trail):
        a = i / len(trail)
        cv2.circle(
            frame,
            (int(tx), int(ty)),
            int(3 + 6 * a),
            color,
            -1
        )

    # Dot + simple glow ring
    cv2.circle(frame, (int(px), int(py)), 14, color, 2)
    cv2.circle(frame, (int(px), int(py)), 6, WHITE, -1)

    cv2.putText(
        frame,
        f"{g_max:.2f} g",
        (cx - 18, cy - radius - 10),
        FONT,
        0.4,
        LIGHT_GRAY,
        1
    )

# ----------------------------------------------------------
# Render loop
# ----------------------------------------------------------
frame_idx = 0
idx = 0  # rolling telemetry index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    video_t = frame_idx / fps
    telemetry_t = video_start + video_t + VIDEO_TIME_OFFSET

    # Manual clamp (faster than np.clip)
    if telemetry_t < telemetry_time[0]:
        telemetry_t = telemetry_time[0]
    elif telemetry_t > telemetry_time[-1]:
        telemetry_t = telemetry_time[-1]

    # Rolling index (faster than searchsorted)
    while idx + 1 < len(telemetry_time) and telemetry_time[idx + 1] <= telemetry_t:
        idx += 1

    raw_long = imu_long[idx]
    raw_lat  = imu_lat[idx]

    if first_sample:
        smooth_long = raw_long
        smooth_lat = raw_lat
        first_sample = False
    else:
        smooth_long = SMOOTH_ALPHA * raw_long + (1 - SMOOTH_ALPHA) * smooth_long
        smooth_lat  = SMOOTH_ALPHA * raw_lat  + (1 - SMOOTH_ALPHA) * smooth_lat

    # Auto scale
    if AUTO_G_MAX:
        g_load = max(abs(smooth_lat), abs(smooth_long))
        target = min(max(g_load * 1.2, G_MAX_MIN), G_MAX_MAX)
        current_g_max = (1 - G_SCALE_RATE) * current_g_max + G_SCALE_RATE * target
    else:
        current_g_max = G_MAX_DEFAULT

    unix_ms = int(telemetry_time[idx] * 1000)

    # HUD
    cv2.putText(frame, f"Unix: {unix_ms}", (40, 50), FONT, 0.7, WHITE, 2)
    cv2.putText(frame, f"Long G: {smooth_long:+.3f}", (40, 100), FONT, 0.9, WHITE, 2)
    cv2.putText(frame, f"Lat G:  {smooth_lat:+.3f}", (40, 150), FONT, 0.9, WHITE, 2)

    draw_g_circle(
        frame,
        lat_g=smooth_lat,
        long_g=smooth_long,
        center=CIRCLE_CENTER,
        radius=CIRCLE_RADIUS,
        g_max=current_g_max,
        trail=trail
    )

    # cv2.imshow("nice", frame)

    # 1 ms is enough, ESC to quit
    # if cv2.waitKey(1) & 0xFF == 27:
        # break

    out.write(frame)
    frame_idx += 1

# ----------------------------------------------------------
# Cleanup
# ----------------------------------------------------------
cap.release()
out.release()

print(f"Saved: {OUTPUT_VIDEO}")
