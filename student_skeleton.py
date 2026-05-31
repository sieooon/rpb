#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


_last_monitor_pts = None
_last_monitor_score = -1.0


def _order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return top_left, top_right, bottom_right, bottom_left


def _quad_area(pts):
    return float(cv2.contourArea(np.asarray(pts, dtype=np.float32).reshape(4, 2)))


def _border_dark_score(gray, quad):
    h, w = gray.shape
    pts = np.asarray(quad, dtype=np.float32)
    scores = []

    for i in range(4):
        p = pts[i]
        q = pts[(i + 1) % 4]
        n = max(20, int(np.linalg.norm(q - p)))
        xs = np.linspace(p[0], q[0], n)
        ys = np.linspace(p[1], q[1], n)
        xi = np.clip(np.round(xs).astype(np.int32), 0, w - 1)
        yi = np.clip(np.round(ys).astype(np.int32), 0, h - 1)
        scores.append(np.mean(gray[yi, xi] < 110))

    return float(np.mean(scores))


def detect_monitor(image):
    """
    TODO: Detect the monitor corners in the input BGR image.

    Return:
      top_left, top_right, bottom_right, bottom_left

    Each point should be an (x, y) pair in the original image coordinate system.
    Return (None, None, None, None) if detection fails.
    """
    global _last_monitor_pts, _last_monitor_score

    if image is None:
        return None, None, None, None

    orig_h, orig_w = image.shape[:2]
    max_work = 700
    scale = 1.0
    if max(orig_h, orig_w) > max_work:
        scale = max_work / float(max(orig_h, orig_w))
        work = cv2.resize(
            image,
            (int(orig_w * scale), int(orig_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = image

    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    image_area = float(h * w)

    best_pts = None
    best_score = -1.0

    def evaluate(mask):
        nonlocal best_pts, best_score

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.01 or area > image_area * 0.85:
                continue

            peri = cv2.arcLength(cnt, True)
            pts = None
            for eps in (0.015, 0.025, 0.04, 0.06):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2)
                    break

            if pts is None:
                rect = cv2.minAreaRect(cnt)
                pts = cv2.boxPoints(rect)

            tl, tr, br, bl = _order_points(pts)
            ordered = np.array([tl, tr, br, bl], dtype=np.float32)
            poly_area = cv2.contourArea(ordered)
            if poly_area < image_area * 0.01 or poly_area > image_area * 0.85:
                continue

            x, y, bw, bh = cv2.boundingRect(ordered.astype(np.int32))
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                if bw > w * 0.9 or bh > h * 0.9:
                    continue

            width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
            height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
            if width < 10 or height < 10:
                continue

            ratio = width / height
            ratio_error = min(abs(ratio - 16.0 / 9.0), abs((1.0 / ratio) - 16.0 / 9.0))
            if ratio_error > 1.7:
                continue

            dark_score = _border_dark_score(gray, ordered)
            if dark_score < 0.12:
                continue

            ratio_score = max(0.1, 1.0 - ratio_error / 1.7)
            score = poly_area * (0.2 + dark_score) * ratio_score
            if score > best_score:
                best_score = score
                best_pts = (tl, tr, br, bl)

    # Thin/dark frame first, then a lightly connected frame mask.
    for thr in (45, 70, 95):
        evaluate(cv2.inRange(gray, 0, thr))

    k = max(3, int(min(h, w) * 0.008))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.inRange(gray, 0, 90)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    evaluate(mask)

    if best_pts is None:
        if _last_monitor_pts is not None:
            return tuple(p.copy() for p in _last_monitor_pts)
        return None, None, None, None

    if scale != 1.0:
        inv = 1.0 / scale
        best_pts = tuple(np.asarray(p, dtype=np.float32) * inv for p in best_pts)

    best_arr = np.asarray(best_pts, dtype=np.float32)
    if _last_monitor_pts is not None:
        last_arr = np.asarray(_last_monitor_pts, dtype=np.float32)
        diag = max(1.0, math.hypot(orig_w, orig_h))
        mean_shift = float(np.mean(np.linalg.norm(best_arr - last_arr, axis=1)) / diag)
        last_area = max(1.0, _quad_area(last_arr))
        area_ratio = _quad_area(best_arr) / last_area

        # In the bag/video the camera and monitor are almost fixed.  If one
        # frame suddenly includes a wrong dark object, keep the stable monitor.
        sudden_jump = mean_shift > 0.08 or area_ratio < 0.72 or area_ratio > 1.38
        if sudden_jump and best_score < _last_monitor_score * 1.8:
            return tuple(p.copy() for p in _last_monitor_pts)

        best_arr = last_arr * 0.82 + best_arr * 0.18
        best_pts = tuple(best_arr[i].astype(np.float32) for i in range(4))

    _last_monitor_pts = tuple(np.asarray(p, dtype=np.float32).copy() for p in best_pts)
    _last_monitor_score = max(best_score, _last_monitor_score * 0.95)

    return best_pts


def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    """
    TODO: Perspective-transform the detected monitor into a front-facing view.

    Return:
      rectified BGR image

    Return None if rectification fails.
    """
    if image is None or any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
        return None

    pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    ordered = list(_order_points(pts))

    lengths = [
        np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])
        for i in range(4)
    ]

    horizontal = 0.5 * (lengths[0] + lengths[2])
    vertical = 0.5 * (lengths[1] + lengths[3])
    if vertical > horizontal:
        ordered = [ordered[1], ordered[2], ordered[3], ordered[0]]
        lengths = lengths[1:] + lengths[:1]

    width = max(1, int(max(lengths[0], lengths[2])))
    width = min(width, 640)
    height = max(1, int(round(width * 9.0 / 16.0)))

    src = np.array(ordered, dtype=np.float32)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, transform, (width, height))


def _extend_line_on_edges(edges, line):
    x1, y1, x2, y2 = line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)
    if length < 1:
        return line

    direction = np.array([dx / length, dy / length], dtype=np.float32)
    base = np.array([float(x1), float(y1)], dtype=np.float32)
    ys, xs = np.where(edges > 0)
    if len(xs) < 2:
        return line

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    rel = pts - base
    proj = rel[:, 0] * direction[0] + rel[:, 1] * direction[1]
    perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])

    h, w = edges.shape
    dist_tol = max(2.0, min(h, w) * 0.008)
    values = np.sort(proj[perp <= dist_tol])
    if len(values) < 2:
        return line

    raw_min = min(0.0, length)
    raw_max = max(0.0, length)
    max_gap = max(5.0, min(h, w) * 0.025)

    best_cluster = None
    best_span = -1.0
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value - prev > max_gap:
            span = prev - start
            if prev >= raw_min - max_gap and start <= raw_max + max_gap and span > best_span:
                best_span = span
                best_cluster = (start, prev)
            start = value
        prev = value

    span = prev - start
    if prev >= raw_min - max_gap and start <= raw_max + max_gap and span > best_span:
        best_span = span
        best_cluster = (start, prev)

    if best_cluster is None or best_span < length * 0.75:
        return line

    start, end = best_cluster
    p1 = base + direction * start
    p2 = base + direction * end
    return (
        int(round(np.clip(p1[0], 0, w - 1))),
        int(round(np.clip(p1[1], 0, h - 1))),
        int(round(np.clip(p2[0], 0, w - 1))),
        int(round(np.clip(p2[1], 0, h - 1))),
    )


def detect_line(rectified):
    """
    TODO: Detect the longest line inside the rectified monitor image.

    Return:
      (x1, y1, x2, y2)

    Return None if line detection fails.
    """
    if rectified is None:
        return None

    h, w = rectified.shape[:2]
    if h < 20 or w < 20:
        return None

    margin_x = max(4, int(w * 0.08))
    margin_y = max(4, int(h * 0.08))
    roi = rectified[margin_y:h - margin_y, margin_x:w - margin_x]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges_gray = cv2.Canny(gray, 25, 90)

    bg = np.median(roi.reshape(-1, 3), axis=0).astype(np.float32)
    dist = np.linalg.norm(roi.astype(np.float32) - bg, axis=2)
    if dist.max() > 1e-6:
        dist = np.clip(dist * (255.0 / dist.max()), 0, 255).astype(np.uint8)
    else:
        dist = np.zeros_like(gray)
    dist = cv2.GaussianBlur(dist, (3, 3), 0)
    edges_color = cv2.Canny(dist, 12, 50)

    edges = cv2.bitwise_or(edges_gray, edges_color)

    min_side = min(h, w)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(8, int(min_side * 0.018)),
        minLineLength=max(12, int(min_side * 0.04)),
        maxLineGap=max(2, int(min_side * 0.006)),
    )
    if lines is None:
        return None

    best = None
    best_len = -1.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        raw = (int(x1), int(y1), int(x2), int(y2))
        extended = _extend_line_on_edges(edges, raw)
        ex1, ey1, ex2, ey2 = extended
        length = math.hypot(float(ex2 - ex1), float(ey2 - ey1))
        if length > best_len:
            best_len = length
            best = extended

    if best is None:
        return None

    x1, y1, x2, y2 = best
    return (x1 + margin_x, y1 + margin_y, x2 + margin_x, y2 + margin_y)


def calculate_angle(line) -> Optional[float]:
    """
    TODO: Calculate the line angle in degrees.

    The angle must be expressed in the rectified monitor coordinate system.
    Return None if the angle cannot be calculated.
    """
    if line is None:
        return None

    x1, y1, x2, y2 = line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    # Problem convention: vertical axis is 0 deg,
    # left-leaning is positive and right-leaning is negative.
    angle = math.degrees(math.atan2(dx, dy))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180

    return angle


class LineDetector(Node):
    def __init__(self) -> None:
        super().__init__("line_detector_node")

        self.declare_parameter("topic_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_student", "/student/angle")

        topic_image = str(self.get_parameter("topic_image").value)
        topic_student = str(self.get_parameter("topic_student").value)

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            topic_image,
            self.image_callback,
            10,
        )

        self.angle_pub = self.create_publisher(
            Float32,
            topic_student,
            10,
        )

        self.line_pub = self.create_publisher(
            Image,
            "/debug/line",
            10,
        )

        self.get_logger().info(
            f"Line detector started. Subscribing to {topic_image!r}, "
            f"publishing to {topic_student!r}."
        )

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc!r}")
            return

        # 1. Detect monitor.
        top_left, top_right, bottom_right, bottom_left = detect_monitor(image)
        if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
            self.get_logger().warning("Monitor not detected.")
            return

        # 2. Rectify monitor.
        rectified = rectify_monitor(image, top_left, top_right, bottom_right, bottom_left)
        if rectified is None:
            self.get_logger().warning("Monitor not rectified.")
            return

        # 3. Detect line.
        line = detect_line(rectified)
        if line is None:
            self.get_logger().warning("Line not detected.")
            return
        self._debug_line(msg, rectified, line)

        # 4. Calculate and publish angle.
        angle = calculate_angle(line)
        if angle is None:
            self.get_logger().warning("Angle not calculated.")
            return

        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)

        self.get_logger().info(f"Line angle: {float(angle):.2f} deg")

    def _debug_line(self, msg, rectified, line) -> None:
        debug_line = rectified.copy()

        x1, y1, x2, y2 = line
        cv2.line(
            debug_line,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            6,
        )

        debug_line_msg = self.bridge.cv2_to_imgmsg(debug_line, encoding="bgr8")
        debug_line_msg.header = msg.header
        self.line_pub.publish(debug_line_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
