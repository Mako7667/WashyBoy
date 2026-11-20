"""
make_markers.py - create a printable A4 sheet of the ArUco markers WashyBoy uses.

  IDs 0-3   table markers: tape near the 4 corners of the work area, inside camera view
  ID 10     cleaning station
  ID 11     clean stack (where cleaned plates are piled)

Output: markers_A4.pdf (print at 100% / "actual size", NOT "fit to page") and markers_A4.png.
After printing, measure one black square with a ruler: it should be 40 mm.
"""

import cv2
import numpy as np
from PIL import Image

from vision import STATION_IDS, TABLE_IDS, _DICT

DPI = 300
MM = DPI / 25.4
MARKER_MM = 40
LABELS = {0: "TABLE 0", 1: "TABLE 1", 2: "TABLE 2", 3: "TABLE 3"}
LABELS.update({k: v.upper().replace("_", " ") for k, v in STATION_IDS.items()})

page = np.full((int(297 * MM), int(210 * MM)), 255, np.uint8)
ids = list(TABLE_IDS) + list(STATION_IDS)
cols, cell_w, cell_h = 2, 95 * MM, 85 * MM
for n, mid in enumerate(ids):
    r, c = divmod(n, cols)
    x0 = int(12 * MM + c * cell_w)
    y0 = int(20 * MM + r * cell_h)
    side = int(MARKER_MM * MM)
    page[y0:y0 + side, x0:x0 + side] = cv2.aruco.generateImageMarker(_DICT, mid, side)
    cv2.putText(page, f"{LABELS[mid]}  (id {mid})", (x0, y0 + side + int(9 * MM)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    cv2.putText(page, "cut leaving a white border", (x0, y0 + side + int(16 * MM)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 120, 2)
cv2.putText(page, f"WashyBoy markers - print at 100% - black squares must measure {MARKER_MM} mm",
            (int(12 * MM), int(10 * MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

img = Image.fromarray(page)
img.save("markers_A4.png", dpi=(DPI, DPI))
img.save("markers_A4.pdf", resolution=DPI)
print("Wrote markers_A4.pdf and markers_A4.png")
