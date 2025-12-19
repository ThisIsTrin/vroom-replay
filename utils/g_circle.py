import cv2
import numpy as np
from config import Config

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
LIGHT_GRAY = (180, 180, 180)


class GCircleRenderer:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.trail = []

    def draw(self, frame, lat_g, long_g, g_max):
        cx, cy = self.center

        cv2.circle(frame, self.center, self.radius, LIGHT_GRAY, 2)
        cv2.line(frame, (cx - self.radius, cy), (cx + self.radius, cy), GRAY, 1)
        cv2.line(frame, (cx, cy - self.radius), (cx, cy + self.radius), GRAY, 1)

        for g in (0.5, 1.0):
            if g < g_max:
                r = int(self.radius * g / g_max)
                cv2.circle(frame, self.center, r, (80, 80, 80), 1)

        lat = np.clip(lat_g, -g_max, g_max)
        lon = np.clip(long_g, -g_max, g_max)

        px = cx + (lat / g_max) * self.radius
        py = cy - (lon / g_max) * self.radius

        g_mag = (lat * lat + lon * lon) ** 0.5

        if g_mag < 0.6:
            color = (0, 180, 0)
        elif g_mag < 1.0:
            color = (0, 200, 220)
        else:
            color = (0, 0, 220)

        self.trail.append((px, py))
        if len(self.trail) > Config.TRAIL_LENGTH:
            self.trail.pop(0)

        for i, (tx, ty) in enumerate(self.trail):
            a = i / len(self.trail)
            cv2.circle(frame, (int(tx), int(ty)), int(3 + 6 * a), color, -1)

        cv2.circle(frame, (int(px), int(py)), 14, color, 2)
        cv2.circle(frame, (int(px), int(py)), 6, WHITE, -1)

        cv2.putText(
            frame,
            f"{g_max:.2f} g",
            (cx - 18, cy - self.radius - 10),
            FONT,
            0.4,
            LIGHT_GRAY,
            1
        )
