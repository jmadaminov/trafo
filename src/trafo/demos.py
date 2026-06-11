"""Interactive demo loops, one per milestone, runnable via `trafo demo <name>`."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .camera import Camera

WINDOW = "trafo"


def camera_demo(camera_index: int = 0) -> None:
    """M1: live preview with FPS counter. Press q or Esc to quit."""
    with Camera(camera_index) as cam:
        print("Camera open. Press q or Esc in the preview window to quit.")
        while True:
            frame, _ = cam.latest()
            if frame is None:
                if cv2.waitKey(30) in (ord("q"), 27):
                    break
                continue
            frame = cv2.flip(frame, 1)  # mirror, so it behaves like a mirror
            cv2.putText(
                frame,
                f"{cam.fps:5.1f} fps  {frame.shape[1]}x{frame.shape[0]}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


def landmarks_demo(camera_index: int = 0) -> None:
    """M2: live eye contours + iris centers. Press q or Esc to quit."""
    from .gaze.landmarks import LEFT_EYE, RIGHT_EYE, FaceLandmarker

    landmarker = FaceLandmarker()
    t0 = time.perf_counter()
    proc_fps = 0.0
    last = t0

    with Camera(camera_index) as cam:
        print("Tracking. Press q or Esc in the preview window to quit.")
        while True:
            frame, ts = cam.latest()
            if frame is None:
                if cv2.waitKey(30) in (ord("q"), 27):
                    break
                continue

            result = landmarker.detect(frame, int((time.perf_counter() - t0) * 1000))

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                proc_fps = 0.9 * proc_fps + 0.1 / dt if proc_fps else 1 / dt

            h, w = frame.shape[:2]
            if result is not None:
                pts = result.pixels(w, h).astype(int)
                for eye in (RIGHT_EYE, LEFT_EYE):
                    cv2.polylines(frame, [pts[eye]], isClosed=True, color=(0, 255, 0), thickness=1)
                for iris in (result.right_iris, result.left_iris):
                    cx, cy = int(iris[0] * w), int(iris[1] * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                # faint full-mesh dots so tracking quality is visible at a glance
                for x, y in pts[::10]:
                    cv2.circle(frame, (x, y), 1, (80, 80, 80), -1)
                status = f"{proc_fps:4.1f} fps (pipeline)"
                color = (0, 255, 0)
            else:
                status = f"{proc_fps:4.1f} fps  NO FACE"
                color = (0, 0, 255)

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) in (ord("q"), 27):
                break

    landmarker.close()
    cv2.destroyAllWindows()


def gaze_demo(camera_index: int = 0) -> None:
    """M3: head-pose axes + raw gaze arrow. Press q or Esc to quit."""
    from .gaze import features, head_pose
    from .gaze.landmarks import LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER, FaceLandmarker

    landmarker = FaceLandmarker()
    t0 = time.perf_counter()

    with Camera(camera_index) as cam:
        print("Tracking. Press q or Esc in the preview window to quit.")
        while True:
            frame, _ = cam.latest()
            if frame is None:
                if cv2.waitKey(30) in (ord("q"), 27):
                    break
                continue

            result = landmarker.detect(frame, int((time.perf_counter() - t0) * 1000))
            h, w = frame.shape[:2]

            if result is not None and result.transform is not None:
                pts = result.pixels(w, h)
                nose = pts[1].astype(int)
                rot = head_pose.rotation(result.transform)
                # Project the head axes orthographically; camera y-up -> image y-down.
                axis_len = 60
                for col, color in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))):
                    d = rot[:, col]
                    end = (int(nose[0] + d[0] * axis_len), int(nose[1] - d[1] * axis_len))
                    cv2.line(frame, tuple(nose), end, color, 2)

                pitch, yaw, roll = head_pose.euler_degrees(result.transform)
                right, left = features.eye_states(result)
                iris = (right.iris_offset + left.iris_offset) / 2

                # Raw gaze arrow: head forward direction nudged by iris offset.
                # Purely a visual sanity check — M4's calibration learns the real mapping.
                fwd = rot[:, 2]
                gaze_dir = np.array([fwd[0] + iris[0] * 4.0, -fwd[1] + iris[1] * 4.0])
                origin = ((pts[RIGHT_IRIS_CENTER] + pts[LEFT_IRIS_CENTER]) / 2).astype(int)
                end = (origin + gaze_dir * 180).astype(int)
                cv2.arrowedLine(frame, tuple(origin), tuple(end), (0, 255, 255), 3, tipLength=0.2)

                blink = features.is_blinking(result)
                status = (
                    f"pitch {pitch:+5.1f}  yaw {yaw:+5.1f}  roll {roll:+5.1f}   "
                    f"iris ({iris[0]:+.3f},{iris[1]:+.3f})" + ("  BLINK" if blink else "")
                )
                color = (0, 255, 0)
            else:
                status = "NO FACE"
                color = (0, 0, 255)

            frame = cv2.flip(frame, 1)
            cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) in (ord("q"), 27):
                break

    landmarker.close()
    cv2.destroyAllWindows()
