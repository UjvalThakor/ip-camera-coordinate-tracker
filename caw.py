"""
Real-Time Camera Coordinate Tracker & Distance Measurement System.
Unified High-Performance Architecture:
  - UDP + MJPEG image2pipe capture with strict boundary verification (zero gray tearing)
  - Zero-delay atomic latest-frame slot with event synchronization (smooth, non-blocking)
  - Two-point P1 / P2 physical measurement engine in native camera resolution
  - Dynamic resolution and rolling monotonic FPS detection
  - Compact glassmorphic transparent HUD overlay directly on video
  - Automatic stream disconnect handling and clean RTSP TEARDOWN
"""
import os
import sys
import time
import math
import enum
import subprocess
import threading
import collections
import numpy as np
import cv2
import imageio_ffmpeg
import argparse

from calibration_manager import CalibrationManager
from distance_tracker import ObjectDistanceTracker
from transparent_hud import TransparentHUD

# Safe UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Camera RTSP Stream Configuration
RTSP_URL = "rtsp://admin:admin123@10.27.1.77:554/Streaming/Channels/101/"
WINDOW_NAME = "Real-Time Camera Coordinate & Distance Tracker (10.27.1.77)"

# Global runtime state for window and mouse callback
current_display_size = (960, 720)
current_native_size = (640, 480)
camera_manager_instance = None


class CameraState(enum.Enum):
    STOPPED = "STOPPED"
    CONNECTING = "CONNECTING"
    STREAM_OPENED = "STREAM_OPENED"
    VALIDATING = "VALIDATING"
    LIVE = "LIVE"
    RECOVERING = "RECOVERING"


class MeasurementManager:
    """
    Manages two-point (P1 -> P2) physical coordinate measurement.
    All stored points strictly use the ORIGINAL NATIVE camera frame coordinates.
    """
    def __init__(self):
        self.p1_orig = None      # (x, y) in original camera coords
        self.p1_time = None      # monotonic timestamp
        self.p2_orig = None      # (x, y) in original camera coords
        self.p2_time = None      # monotonic timestamp

    def add_click(self, orig_x, orig_y, click_time):
        """State machine for P1/P2 selection."""
        if self.p1_orig is None:
            # Set Point 1
            self.p1_orig = (orig_x, orig_y)
            self.p1_time = click_time
            self.p2_orig = None
            self.p2_time = None
            return "P1_SET"
        elif self.p2_orig is None:
            # Set Point 2
            self.p2_orig = (orig_x, orig_y)
            self.p2_time = click_time
            return "P2_SET"
        else:
            # Cycle restart: new P1
            self.p1_orig = (orig_x, orig_y)
            self.p1_time = click_time
            self.p2_orig = None
            self.p2_time = None
            return "P1_RESET"

    def reset(self):
        """Clears both points cleanly."""
        self.p1_orig = None
        self.p1_time = None
        self.p2_orig = None
        self.p2_time = None

    def get_summary(self, distance_tracker, frame_w, frame_h):
        """Calculates dx, dy, pixel distance, dt, and real-world metric distance."""
        if self.p1_orig is None or self.p2_orig is None:
            return None
        dx = abs(self.p2_orig[0] - self.p1_orig[0])
        dy = abs(self.p2_orig[1] - self.p1_orig[1])
        dt = abs(self.p2_time - self.p1_time) if (self.p1_time and self.p2_time) else 0.0
        metric_dist, px_dist = distance_tracker.calculate_metric_distance(self.p1_orig, self.p2_orig, frame_w, frame_h)
        return {
            "dx": dx,
            "dy": dy,
            "pixel_dist": px_dist,
            "dt": dt,
            "metric_dist": metric_dist,
        }


class AtomicLatestFrameSlot:
    """
    Lock-protected atomic latest frame slot with event synchronization.
    The capture thread continuously pushes the newest decoded frame.
    The display loop waits on the frame event for zero latency and no busy-polling.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self._latest_frame = None
        self._frame_id = 0
        self._capture_ts = 0.0
        self._consumed_id = -1

        # Dynamic stream dimensions
        self.native_width = 640
        self.native_height = 480

        # Rolling capture FPS calculation
        self._capture_timestamps = collections.deque(maxlen=30)
        self.capture_fps = 0.0

    def publish_frame(self, frame, frame_id, capture_ts):
        """Publishes newest frame from the background capture worker."""
        h, w = frame.shape[:2]
        now = time.perf_counter()

        with self.lock:
            self._latest_frame = frame
            self._frame_id = frame_id
            self._capture_ts = capture_ts
            self.native_width = w
            self.native_height = h

            self._capture_timestamps.append(now)
            if len(self._capture_timestamps) > 1:
                self.capture_fps = (len(self._capture_timestamps) - 1) / (
                    self._capture_timestamps[-1] - self._capture_timestamps[0]
                )

        self.new_frame_event.set()
        return True

    def get_display_frame(self, timeout=0.033):
        """
        Retrieves newest frame for display loop.
        Waits for frame arrival event up to timeout to prevent busy-loop stutter.
        """
        self.new_frame_event.wait(timeout=timeout)
        self.new_frame_event.clear()

        with self.lock:
            if self._latest_frame is None:
                return False, None, -1, 0.0, False, self.capture_fps, self.native_width, self.native_height

            is_new = (self._frame_id != self._consumed_id)
            self._consumed_id = self._frame_id
            frame_copy = self._latest_frame.copy()
            fid = self._frame_id
            cts = self._capture_ts
            cap_fps = self.capture_fps
            nw = self.native_width
            nh = self.native_height

        return True, frame_copy, fid, cts, is_new, cap_fps, nw, nh

    def clear(self):
        with self.lock:
            self._latest_frame = None
            self._frame_id = 0
            self._consumed_id = -1
            self._capture_timestamps.clear()
            self.capture_fps = 0.0
            self.new_frame_event.clear()


def is_clean_frame(frame):
    """Validates that a frame is non-null, 3-channel, and not completely blank."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0 or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    if frame.std() < 0.5:
        return False
    return True


class RTSPCaptureManager:
    """
    Dedicated RTSP Capture Manager with Strict UDP Framing and Cooldown Handling.
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.state = CameraState.STOPPED
        self.lifecycle_lock = threading.Lock()
        self._live_event = threading.Event()
        self._stop_event = threading.Event()

        self.proc = None
        self.capture_thread = None
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.slot = AtomicLatestFrameSlot()

        self.width = None
        self.height = None

    @property
    def native_width(self):
        return self.slot.native_width if self.width is None else self.width

    @property
    def native_height(self):
        return self.slot.native_height if self.height is None else self.height

    def start(self):
        with self.lifecycle_lock:
            if self.state != CameraState.STOPPED:
                return False
            self._stop_event.clear()
            self._live_event.clear()
            self.slot.clear()
            self.state = CameraState.CONNECTING
            print("[CAMERA] Starting capture pipeline", flush=True)
            self.capture_thread = threading.Thread(
                target=self._capture_worker,
                name="RTSPCaptureThread",
                daemon=True
            )
            self.capture_thread.start()
            return True

    def wait_until_live(self, timeout=30.0):
        return self._live_event.wait(timeout=timeout)

    def _drain_stderr(self, proc):
        """Continuously drains stderr to avoid pipe buffer deadlocks on Windows."""
        try:
            while not self._stop_event.is_set() and proc.poll() is None:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
        except Exception:
            pass

    def _safe_close_proc(self):
        """Safely terminates stream process with graceful RTSP TEARDOWN, then closes pipes."""
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write(b"q\n")
                    self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=0.5)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
            try:
                if self.proc.stdout:
                    self.proc.stdout.close()
            except Exception:
                pass
            try:
                if self.proc.stderr:
                    self.proc.stderr.close()
            except Exception:
                pass
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            self.proc = None

    def _capture_worker(self):
        """
        Pure background capture thread.
        Uses UDP transport with high-throughput MJPEG image2pipe.
        Strict JPEG delimiter inspection discards partial packets, eliminating gray smears.
        """
        CONSECUTIVE_VALID_REQUIRED = 3
        MAX_READ_FAILURES = 25

        cmd = [
            self.ffmpeg_exe,
            "-loglevel", "error",
            "-rtsp_transport", "udp",
            "-buffer_size", "26214400",
            "-max_delay", "500000",
            "-reorder_queue_size", "4000",
            "-i", self.rtsp_url,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "2",
            "-"
        ]

        is_recovering = False
        seq_id = 0

        while not self._stop_event.is_set():
            if not is_recovering:
                self.state = CameraState.CONNECTING
            print("[CAMERA] Opening RTSP stream...", flush=True)

            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10**6
                )
                threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()
            except Exception as e:
                if self._stop_event.is_set():
                    break
                print(f"[CAMERA] VideoCapture open failed: {e}", flush=True)
                time.sleep(1.0)
                continue

            self.state = CameraState.VALIDATING
            consecutive_good = 0
            validation_attempts = 0
            validated = False
            pipe_buf = bytearray()
            t_val_start = time.perf_counter()

            # Startup Validation Loop: Require 3 consecutive clean frames (up to 6s per attempt)
            while not self._stop_event.is_set() and self.proc.poll() is None and validation_attempts < 100 and (time.perf_counter() - t_val_start < 6.0):
                try:
                    chunk = self.proc.stdout.read1(16384)
                except Exception:
                    break
                if not chunk:
                    if self._stop_event.is_set() or self.proc.poll() is not None:
                        break
                    time.sleep(0.005)
                    continue
                pipe_buf.extend(chunk)

                # Extract and validate complete JPEG frames
                while True:
                    soi = pipe_buf.find(b"\xff\xd8")
                    if soi == -1:
                        if len(pipe_buf) > 1:
                            pipe_buf = pipe_buf[-1:]
                        break
                    eoi = pipe_buf.find(b"\xff\xd9", soi + 2)
                    if eoi == -1:
                        if soi > 0:
                            pipe_buf = pipe_buf[soi:]
                        break

                    # Strict delimiter check: discard incomplete frames where next SOI comes before EOI
                    next_soi = pipe_buf.find(b"\xff\xd8", soi + 2)
                    if next_soi != -1 and next_soi < eoi:
                        pipe_buf = pipe_buf[next_soi:]
                        continue

                    jpg_bytes = pipe_buf[soi:eoi + 2]
                    pipe_buf = pipe_buf[eoi + 2:]

                    validation_attempts += 1
                    seq_id += 1
                    t_cap = time.perf_counter()

                    raw_frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if is_clean_frame(raw_frame):
                        consecutive_good += 1
                        if self.width is None:
                            self.height, self.width = raw_frame.shape[:2]

                        if consecutive_good >= CONSECUTIVE_VALID_REQUIRED:
                            self.slot.publish_frame(raw_frame, seq_id, t_cap)
                            self.state = CameraState.LIVE
                            self._live_event.set()
                            if is_recovering:
                                print("[CAMERA] Stream recovered successfully", flush=True)
                                is_recovering = False
                            print("[CAMERA] LIVE stream locked on", flush=True)
                            validated = True
                            break
                    else:
                        consecutive_good = 0

                if validated:
                    break

            if not validated:
                if self._stop_event.is_set():
                    break
                print("[CAMERA] Stream syncing (socket cooldown 2.5s)...", flush=True)
                self._safe_close_proc()
                time.sleep(2.5)
                continue

            # LIVE Streaming Loop: Read chunk -> strict JPEG extract -> decode -> publish
            consecutive_failures = 0

            while not self._stop_event.is_set() and self.proc.poll() is None:
                try:
                    chunk = self.proc.stdout.read1(16384)
                except Exception:
                    break
                if not chunk:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_READ_FAILURES or self.proc.poll() is not None:
                        break
                    time.sleep(0.002)
                    continue

                pipe_buf.extend(chunk)

                while True:
                    soi = pipe_buf.find(b"\xff\xd8")
                    if soi == -1:
                        if len(pipe_buf) > 1:
                            pipe_buf = pipe_buf[-1:]
                        break
                    eoi = pipe_buf.find(b"\xff\xd9", soi + 2)
                    if eoi == -1:
                        if soi > 0:
                            pipe_buf = pipe_buf[soi:]
                        break

                    # Strict boundary guard against partial packets
                    next_soi = pipe_buf.find(b"\xff\xd8", soi + 2)
                    if next_soi != -1 and next_soi < eoi:
                        pipe_buf = pipe_buf[next_soi:]
                        continue

                    jpg_bytes = pipe_buf[soi:eoi + 2]
                    pipe_buf = pipe_buf[eoi + 2:]

                    seq_id += 1
                    t_cap = time.perf_counter()
                    raw_frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if is_clean_frame(raw_frame):
                        consecutive_failures = 0
                        self.slot.publish_frame(raw_frame, seq_id, t_cap)

            # Auto-reconnection on stream loss
            if self._stop_event.is_set():
                break

            print("[CAMERA] Stream connection interrupted. Reconnecting...", flush=True)
            self.state = CameraState.RECOVERING
            self._live_event.clear()
            is_recovering = True
            self._safe_close_proc()
            time.sleep(1.5)

        self._safe_close_proc()
        self.state = CameraState.STOPPED
        print("[CAMERA] Capture thread exited cleanly", flush=True)

    def get_display_snapshot(self, timeout=0.033):
        return self.slot.get_display_frame(timeout=timeout)

    def shutdown(self):
        with self.lifecycle_lock:
            if self.state == CameraState.STOPPED and self.capture_thread is None:
                return
            self._stop_event.set()
            self._safe_close_proc()
            if self.capture_thread is not None and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
                self.capture_thread = None
            self.slot.clear()
            self.state = CameraState.STOPPED
            print("[CAMERA] Shutdown complete", flush=True)


# Global measurement manager instance
measurement_mgr = MeasurementManager()


def on_mouse_click(event, x, y, flags, param):
    """
    OpenCV Mouse Callback:
    Maps click coordinates from display window space back to ORIGINAL CAMERA FRAME coordinates.
    Points are recorded in native coordinates so they remain completely invariant to window scaling.
    """
    global measurement_mgr, current_display_size, current_native_size

    if event == cv2.EVENT_LBUTTONDOWN:
        disp_w, disp_h = current_display_size
        nat_w, nat_h = current_native_size

        if disp_w <= 0 or disp_h <= 0:
            return

        # Map display coordinate (x, y) to original camera coordinate
        orig_x = int(round(x * (nat_w / float(disp_w))))
        orig_y = int(round(y * (nat_h / float(disp_h))))

        # Clamp within frame dimensions
        orig_x = max(0, min(orig_x, nat_w - 1))
        orig_y = max(0, min(orig_y, nat_h - 1))

        t_click = time.perf_counter()
        action = measurement_mgr.add_click(orig_x, orig_y, t_click)

        if action == "P1_SET":
            print(f"[📍 P1 SET] Native: ({orig_x}, {orig_y}) | Display: ({x}, {y})", flush=True)
        elif action == "P2_SET":
            print(f"[📍 P2 SET] Native: ({orig_x}, {orig_y}) | Display: ({x}, {y})", flush=True)
        elif action == "P1_RESET":
            print(f"[📍 RESTART] New P1 Native: ({orig_x}, {orig_y})", flush=True)


def run_camera():
    global camera_manager_instance, current_display_size, current_native_size, measurement_mgr

    parser = argparse.ArgumentParser(description="Real-Time Object Distance & Approach Speed Measurement")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode without physical camera")
    parser.add_argument("--marker-size", type=float, default=0.100, help="Physical marker side length in meters (default: 0.100)")
    parser.add_argument("--marker-id", type=int, default=0, help="Target ArUco marker ID (default: 0)")
    parser.add_argument("--calibration", type=str, default="camera_calibration.json", help="Path to camera_calibration.json")
    parser.add_argument("--target", type=str, default="auto", choices=["auto", "page", "aruco"], help="Target object: 'auto' (detects either), 'page' (paper page), 'aruco' (ArUco marker)")
    parser.add_argument("--page-width", type=float, default=0.210, help="Physical width of paper page in meters (default: 0.210m = 21cm for A4)")
    parser.add_argument("--page-height", type=float, default=0.297, help="Physical height of paper page in meters (default: 0.297m = 29.7cm for A4)")
    args = parser.parse_args()

    print("=" * 65)
    print(" Real-Time Camera Coordinate & Distance Tracker")
    print(f" Mode: {'SIMULATION' if args.sim else 'LIVE RTSP STREAM'}")
    if not args.sim:
        print(f" RTSP URL: {RTSP_URL}")
    print(f" Target Mode: {args.target.upper()} | Paper: {args.page_width*100:.1f}x{args.page_height*100:.1f} cm | Marker: {args.marker_size * 100.0:.1f} cm (#{args.marker_id})")
    print(" Controls:")
    print("   [Left-Click] : Set P1 (1st click) -> Set P2 (2nd click)")
    print("   [C] or [R]   : Clear / Reset P1 and P2 measurement")
    print("   [P] Key      : Toggle Target Object mode (AUTO <-> PAGE <-> ARUCO)")
    print("   [M] Key      : Toggle transparent HUD overlay")
    print("   [S] Key      : Save snapshot image with measurement overlay")
    print("   [Q] or [ESC] : Quit cleanly")
    print("=" * 65)

    # Initialize Calibration Manager & Distance Tracker
    calib_mgr = CalibrationManager(args.calibration, default_marker_size=args.marker_size)
    calib_mgr.marker_id = args.marker_id
    distance_tracker = ObjectDistanceTracker(
        calib_mgr,
        marker_size_m=args.marker_size,
        marker_id=args.marker_id,
        deadband_m=0.005,
        smoothing_alpha=0.35,
    )
    distance_tracker.target_mode = args.target.upper()
    distance_tracker.page_width_m = args.page_width
    distance_tracker.page_height_m = args.page_height
    hud = TransparentHUD()
    show_hud = True

    sim_gen = None
    if args.sim:
        from test_object_distance import SyntheticTargetGenerator
        nat_w, nat_h = 640, 480
        sim_gen = SyntheticTargetGenerator(width=nat_w, height=nat_h, marker_size_m=args.marker_size, marker_id=args.marker_id)
        calib_mgr.set_calibration_data(sim_gen.K, sim_gen.D, nat_w, nat_h, marker_size=args.marker_size)
        print("\n✅ Simulation stream initialized. Camera moving smoothly toward/away from object.\n", flush=True)
    else:
        camera_manager_instance = RTSPCaptureManager(RTSP_URL)
        camera_manager_instance.start()

        print("\n[+] Synchronizing live camera feed...", end="", flush=True)
        t_start_sync = time.perf_counter()

        is_live = camera_manager_instance.wait_until_live(timeout=35.0)
        if not is_live:
            print("\n[!] Failed to connect to camera. Shutting down.", flush=True)
            camera_manager_instance.shutdown()
            return

        print(f"\n✅ Camera connected in {time.perf_counter() - t_start_sync:.2f}s! Crystal-clear video window opened.\n", flush=True)

    window_created = False
    display_timestamps = collections.deque(maxlen=30)
    display_fps = 0.0
    last_log_time = time.perf_counter()
    last_valid_frame_time = time.perf_counter()
    last_known_frame = None

    try:
        while True:
            t_cycle_start = time.perf_counter()

            if args.sim:
                # Smooth cyclical 3D motion: 2.20m to 0.80m
                cycle = (t_cycle_start * 0.8) % (2.0 * math.pi)
                sim_dist = 1.50 + 0.70 * math.cos(cycle)
                sim_yaw = 4.0 * math.sin(cycle * 2.0)
                frame = sim_gen.render_frame_at_distance(sim_dist, yaw_deg=sim_yaw)
                cts = t_cycle_start
                is_new = True
                nat_w, nat_h = 640, 480
                time.sleep(0.025)
            else:
                ret, frame, fid, cts, is_new, cap_fps, nat_w, nat_h = camera_manager_instance.get_display_snapshot(timeout=0.033)

                if ret and frame is not None:
                    last_valid_frame_time = t_cycle_start
                    last_known_frame = frame
                else:
                    # Stream lost check
                    if t_cycle_start - last_valid_frame_time > 2.5:
                        # Display Stream Lost card on frozen frame or dark screen
                        base_img = last_known_frame if last_known_frame is not None else np.full((480, 640, 3), 30, dtype=np.uint8)
                        disp_img = cv2.resize(base_img, current_display_size)
                        cv2.putText(disp_img, "STREAM INTERRUPTED - RECONNECTING...", (disp_img.shape[1]//2 - 220, disp_img.shape[0]//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 140, 255), 2, cv2.LINE_AA)
                        cv2.imshow(WINDOW_NAME, disp_img)
                        key = cv2.waitKey(20) & 0xFF
                        if key == ord("q") or key == 27:
                            break
                        continue
                    time.sleep(0.002)
                    continue

            # Rolling display FPS calculation
            if is_new:
                display_timestamps.append(t_cycle_start)
                if len(display_timestamps) > 1:
                    display_fps = (len(display_timestamps) - 1) / (
                        display_timestamps[-1] - display_timestamps[0]
                    )

            # Update global native resolution dynamically
            current_native_size = (nat_w, nat_h)

            # Calculate display dimensions preserving native camera aspect ratio
            target_h = 720
            target_w = int(round(target_h * (nat_w / float(nat_h))))
            current_display_size = (target_w, target_h)

            # Process object distance tracking (ArUco / Paper Page)
            telemetry = distance_tracker.process_frame(frame, timestamp=cts)

            # Compute P1 / P2 measurement summary in original frame space
            meas_summary = measurement_mgr.get_summary(distance_tracker, nat_w, nat_h)

            # Periodic console log (every 2.5s)
            if t_cycle_start - last_log_time >= 2.5:
                p1_info = f"P1={measurement_mgr.p1_orig}" if measurement_mgr.p1_orig else "P1=--"
                p2_info = f"P2={measurement_mgr.p2_orig}" if measurement_mgr.p2_orig else "P2=--"
                m_info = f"Dist={meas_summary['pixel_dist']:.1f}px ({meas_summary['metric_dist']:.2f}m)" if meas_summary else "Dist=--"
                t_info = f"Target={telemetry['distance_m']:.2f}m ({telemetry['speed_mps']:.2f}m/s)" if telemetry['marker_detected'] else "Target=LOST"
                print(f"[STATUS] {p1_info} | {p2_info} | {m_info} | {t_info} | FPS: {display_fps:.1f}", flush=True)
                last_log_time = t_cycle_start

            # Resize native frame to display window dimensions
            display_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Project coordinates from native camera space to display window space
            scale_x = target_w / float(nat_w)
            scale_y = target_h / float(nat_h)

            p1_data = None
            if measurement_mgr.p1_orig is not None:
                p1_data = {
                    "orig": measurement_mgr.p1_orig,
                    "disp": (int(round(measurement_mgr.p1_orig[0] * scale_x)), int(round(measurement_mgr.p1_orig[1] * scale_y))),
                    "time": measurement_mgr.p1_time,
                }

            p2_data = None
            if measurement_mgr.p2_orig is not None:
                p2_data = {
                    "orig": measurement_mgr.p2_orig,
                    "disp": (int(round(measurement_mgr.p2_orig[0] * scale_x)), int(round(measurement_mgr.p2_orig[1] * scale_y))),
                    "time": measurement_mgr.p2_time,
                }

            # Scale detected target annotations to display space
            disp_telemetry = None
            if telemetry is not None:
                disp_telemetry = telemetry.copy()
                if telemetry["corners"] is not None:
                    scaled_c = telemetry["corners"].copy().astype(np.float32).reshape(4, 2)
                    scaled_c[:, 0] *= scale_x
                    scaled_c[:, 1] *= scale_y
                    disp_telemetry["corners"] = scaled_c
                if telemetry["center_pixel"] is not None:
                    cx, cy = telemetry["center_pixel"]
                    disp_telemetry["center_pixel"] = (int(round(cx * scale_x)), int(round(cy * scale_y)))

            # Render Unified Transparent HUD directly on the frame in a single pass
            if show_hud:
                display_frame = hud.render_overlay(
                    display_frame,
                    p1_data,
                    p2_data,
                    meas_summary,
                    telemetry=disp_telemetry,
                    fps=display_fps,
                    native_w=nat_w,
                    native_h=nat_h,
                    is_live=True
                )

            # Create window with auto-sizing and mouse callback
            if not window_created:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)
                window_created = True

            cv2.imshow(WINDOW_NAME, display_frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("\n[+] Exit requested by user.", flush=True)
                break
            elif key == ord("c") or key == ord("C") or key == ord("r") or key == ord("R"):
                measurement_mgr.reset()
                print("[+] P1 / P2 measurement reset.", flush=True)
            elif key == ord("p") or key == ord("P"):
                # Cycle Target Object mode: AUTO -> PAGE -> ARUCO -> AUTO
                modes = ["AUTO", "PAGE", "ARUCO"]
                cur_idx = modes.index(distance_tracker.target_mode) if distance_tracker.target_mode in modes else 0
                new_mode = modes[(cur_idx + 1) % len(modes)]
                distance_tracker.set_target_mode(new_mode)
            elif key == ord("m") or key == ord("M"):
                show_hud = not show_hud
                print(f"[HUD] Overlay display: {'ENABLED' if show_hud else 'DISABLED'}", flush=True)
            elif key == ord("s") or key == ord("S"):
                os.makedirs("snapshots", exist_ok=True)
                snap_fn = f"snapshots/snapshot_{int(time.time())}.png"
                cv2.imwrite(snap_fn, display_frame)
                print(f"📸 Snapshot saved: '{snap_fn}'", flush=True)

    except KeyboardInterrupt:
        print("\n[+] Interrupted by user.", flush=True)
    finally:
        cv2.destroyAllWindows()
        if camera_manager_instance is not None:
            camera_manager_instance.shutdown()
        print("✅ Live stream closed cleanly.\n", flush=True)


if __name__ == "__main__":
    run_camera()
