import cv2
import numpy as np
import time

from config import Config
from telemetry import TelemetryData
from utils.g_circle import GCircleRenderer

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)


class TelemetryVideoRenderer:
    def __init__(self):
        self.cfg = Config
        self.telemetry = TelemetryData(self.cfg.CSV_PATH)

        self.cap = cv2.VideoCapture(self.cfg.VIDEO_PATH)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_cnt = self.total_frames
        self.video_duration = frame_cnt / self.fps
        self.video_start = self.telemetry.time[-1] - self.video_duration

        self.out = cv2.VideoWriter(
            self.cfg.OUTPUT_VIDEO,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )

        self.g_circle = GCircleRenderer(
            center=(self.width - 160, self.height - 160),
            radius=self.cfg.CIRCLE_RADIUS
        )

        self.idx = 0
        self.smooth_lat = 0.0
        self.smooth_long = 0.0
        self.first_sample = True
        self.current_g_max = self.cfg.G_MAX_DEFAULT

        # ---- render status ----
        self.start_time = time.time()

    def _update_index(self, t):
        while (
            self.idx + 1 < len(self.telemetry.time)
            and self.telemetry.time[self.idx + 1] <= t
        ):
            self.idx += 1

    def render(self):
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            video_t = frame_idx / self.fps
            telemetry_t = (
                self.video_start
                + video_t
                + self.cfg.VIDEO_TIME_OFFSET
            )
            telemetry_t = self.telemetry.clamp_time(telemetry_t)
            self._update_index(telemetry_t)

            raw_long = self.telemetry.long_g[self.idx]
            raw_lat = self.telemetry.lat_g[self.idx]

            if self.first_sample:
                self.smooth_long = raw_long
                self.smooth_lat = raw_lat
                self.first_sample = False
            else:
                a = self.cfg.SMOOTH_ALPHA
                self.smooth_long = a * raw_long + (1 - a) * self.smooth_long
                self.smooth_lat = a * raw_lat + (1 - a) * self.smooth_lat

            # Auto G scaling
            if self.cfg.AUTO_G_MAX:
                g_load = max(abs(self.smooth_lat), abs(self.smooth_long))
                target = np.clip(
                    g_load * 1.2,
                    self.cfg.G_MAX_MIN,
                    self.cfg.G_MAX_MAX
                )
                self.current_g_max = (
                    (1 - self.cfg.G_SCALE_RATE) * self.current_g_max
                    + self.cfg.G_SCALE_RATE * target
                )

            unix_ms = int(self.telemetry.time[self.idx] * 1000)

            # HUD
            cv2.putText(frame, f"Unix: {unix_ms}", (40, 50), FONT, 0.7, WHITE, 2)
            cv2.putText(frame, f"Long G: {self.smooth_long:+.3f}", (40, 100), FONT, 0.9, WHITE, 2)
            cv2.putText(frame, f"Lat G:  {self.smooth_lat:+.3f}", (40, 150), FONT, 0.9, WHITE, 2)

            self.g_circle.draw(
                frame,
                lat_g=self.smooth_lat,
                long_g=self.smooth_long,
                g_max=self.current_g_max
            )

            self.out.write(frame)
            frame_idx += 1

            # ---------- render status (Option A) ----------
            if frame_idx % int(self.fps) == 0:
                elapsed = time.time() - self.start_time
                progress = frame_idx / self.total_frames * 100

                fps_now = frame_idx / elapsed if elapsed > 0 else 0
                eta = (
                    (self.total_frames - frame_idx) / fps_now
                    if fps_now > 0 else 0
                )

                print(
                    f"\rRendering: {progress:6.2f}% | "
                    f"{frame_idx}/{self.total_frames} frames | "
                    f"{fps_now:5.1f} fps | "
                    f"ETA: {eta:6.1f}s",
                    end="",
                    flush=True
                )

        print()  # newline after progress bar

        self.cap.release()
        self.out.release()
        print(f"Saved: {self.cfg.OUTPUT_VIDEO}")
