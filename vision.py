"""
vision.py - camera perception for WashyBoy, built on OpenCV (ArUco markers + contour detection).

How it works
------------
* Four ArUco markers (IDs 0-3) are taped to the table. Their positions in the ARM's
  coordinate frame are measured once (setup_wizard.py: touch each with the gripper tip)
  and stored in table_markers.json.
* Every frame, the markers are found again and a homography (pixel -> table metres) is
  recomputed. If the camera gets bumped, it re-aligns itself automatically.
* Station markers mark places the robot should use:  ID 10 = cleaning station,
  ID 11 = clean stack. Move the paper marker and the robot follows it.
* Plates are found by shape: square outlines about 70 mm wide (your printed plate).
  Dishes are also found as circles. Each detection is returned in table metres with an
  angle, so the gripper can line up with the plate.

Print the markers with:  python make_markers.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TABLE_IDS = (0, 1, 2, 3)
STATION_IDS = {10: "clean_station", 11: "stack"}
PLATE_SIDE_M = 0.070
MARKERS_FILE = Path(__file__).with_name("table_markers.json")

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_DETECTOR = cv2.aruco.ArucoDetector(_DICT, cv2.aruco.DetectorParameters())


@dataclass
class Plate:
    x: float  # metres, arm frame
    y: float
    yaw_deg: float  # rotation of the square, folded into [-45, 45)
    side_m: float  # measured side length (sanity check, ~0.070)
    px: tuple[int, int]  # centre in the image


@dataclass
class Scene:
    plates: list[Plate]
    dishes: list[tuple[float, float, float]]  # (x, y, radius) metres
    stations: dict[str, tuple[float, float]]  # name -> (x, y) metres
    markers_px: dict[int, tuple[float, float]]
    homography: np.ndarray | None
    image: np.ndarray  # annotated BGR image for display


def load_table_markers() -> dict[int, tuple[float, float]]:
    if not MARKERS_FILE.exists():
        return {}
    return {int(k): tuple(v) for k, v in json.loads(MARKERS_FILE.read_text()).items()}


def save_table_markers(m: dict[int, tuple[float, float]]):
    MARKERS_FILE.write_text(json.dumps({str(k): list(v) for k, v in m.items()}, indent=2))


def detect_markers(rgb: np.ndarray):
    """Returns ({id: centre_px}, corners, ids) for every ArUco marker in view."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None:
        return {}, [], None
    return {int(i): tuple(c[0].mean(axis=0)) for i, c in zip(ids.flatten(), corners)}, corners, ids


def _to_table(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts_px.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def analyse(rgb: np.ndarray, table_markers: dict[int, tuple[float, float]] | None = None) -> Scene:
    """Find markers, plates and dishes in one RGB frame."""
    table_markers = load_table_markers() if table_markers is None else table_markers
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    markers_px, corners, ids = detect_markers(rgb)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    # pixel -> table homography from whichever table markers are visible (need 4)
    H = None
    common = [i for i in TABLE_IDS if i in markers_px and i in table_markers]
    if len(common) == 4:
        H, _ = cv2.findHomography(
            np.array([markers_px[i] for i in common]), np.array([table_markers[i] for i in common])
        )
    elif len(common) == 3:  # one marker covered (e.g. by a plate): affine fallback, slightly less exact
        A = cv2.getAffineTransform(np.float32([markers_px[i] for i in common]),
                                   np.float32([table_markers[i] for i in common]))
        H = np.vstack([A, [0, 0, 1]])

    stations = {}
    if H is not None:
        for mid, name in STATION_IDS.items():
            if mid in markers_px:
                stations[name] = tuple(_to_table(H, np.array([markers_px[mid]]))[0])

    # mask out markers so their black squares are not mistaken for plates
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(gray)
    for c in corners:
        cv2.fillConvexPoly(mask, c[0].astype(np.int32), 255)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))

    plates = _find_plates(gray, mask, H, vis) if H is not None else []
    dishes = _find_dishes(gray, H, vis) if H is not None else []

    if H is None:
        cv2.putText(vis, f"table markers seen: {len(common)}/4", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for mid, name in STATION_IDS.items():
        if name in stations:
            cv2.putText(vis, name, tuple(int(v) for v in markers_px[mid]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return Scene(plates, dishes, stations, markers_px, H, vis)


def _find_plates(gray, marker_mask, H, vis) -> list[Plate]:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    edges[marker_mask > 0] = 0
    # RETR_LIST, not EXTERNAL: the table's own outline would otherwise hide everything inside it
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    plates: list[Plate] = []
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        rect = cv2.minAreaRect(c)
        box_px = cv2.boxPoints(rect)
        box_m = _to_table(H, box_px)
        sides = [np.linalg.norm(box_m[i] - box_m[(i + 1) % 4]) for i in range(4)]
        side = float(np.mean(sides))
        squareness = min(sides) / max(sides)
        fill = cv2.contourArea(c) / max(rect[1][0] * rect[1][1], 1)
        if not (0.8 * PLATE_SIDE_M < side < 1.25 * PLATE_SIDE_M and squareness > 0.85 and fill > 0.75):
            continue
        cx, cy = box_m.mean(axis=0)
        # each outline yields an inner and an outer contour; keep one per plate
        if any(np.hypot(p.x - cx, p.y - cy) < PLATE_SIDE_M / 3 for p in plates):
            continue
        edge = box_m[1] - box_m[0]
        yaw = (np.degrees(np.arctan2(edge[1], edge[0])) + 45) % 90 - 45
        centre_px = tuple(int(v) for v in box_px.mean(axis=0))
        plates.append(Plate(float(cx), float(cy), float(yaw), side, centre_px))
        cv2.drawContours(vis, [box_px.astype(np.int32)], 0, (0, 255, 0), 2)
        cv2.putText(vis, f"plate {cx:.3f},{cy:.3f} {yaw:+.0f}deg", (centre_px[0] - 80, centre_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return plates


def _find_dishes(gray, H, vis) -> list[tuple[float, float, float]]:
    circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=100, param2=45, minRadius=15, maxRadius=200)
    out = []
    if circles is None:
        return out
    for u, v, r in circles[0]:
        centre, rim = _to_table(H, np.array([[u, v], [u + r, v]]))
        out.append((float(centre[0]), float(centre[1]), float(np.linalg.norm(rim - centre))))
        cv2.circle(vis, (int(u), int(v)), int(r), (255, 200, 0), 1)
    return out
