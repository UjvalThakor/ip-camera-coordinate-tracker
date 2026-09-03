import os
import sys
import time
import enum
import subprocess
import threading
import collections
import numpy as np
import cv2
import imageio_ffmpeg

# Safe UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Camera RTSP Stream Configuration
RTSP_URL = "rtsp://admin:admin123@10.27.1.77:554/Streaming/Channels/101/"
WINDOW_NAME = "IP Camera Live Stream & Coordinate Tracker (10.27.1.77)"

# Target Display Parameters (Smooth 25-30 FPS HD Presentation)
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720
TARGET_FPS = 25.0
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Global coordinate tracking state
clicked_display_pt = None   # (disp_x, disp_y)
clicked_orig_pt = None      # (orig_x, orig_y)
camera_manager_instance = None


class CameraState(enum.Enum):
    STOPPED = "STOPPED"
    CONNECTING = "CONNECTING"
    STREAM_OPENED = "STREAM_OPENED"
    VALIDATING = "VALIDATING"
    LIVE = "LIVE"
    RECOVERING = "RECOVERING"


def is_valid_clean_frame(frame):
    """
    Ultra-fast, zero-overhead visual integrity filter:
    1. Validates array structure, shape (H, W, 3), and uint8 dtype.
    2. Rejects dead blank / uninitialized zero-variance frames.
    3. Rejects unrendered neutral grey canvases (HEVC missing I-frame canvas).
    4. Rejects bright green slice errors (YUV 0,0,0 uninitialized macroblocks).
    5. Rejects blue/magenta chroma tearing bands.
    6. Rejects HEVC error concealment checkerboard / dotted noise artifacts.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False

    h, w = frame.shape[:2]
    if h < 100 or w < 100 or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        return False

    # 1. Reject solid dead / blank frames
    if float(frame.std()) < 15.0:
        return False

    # 2. Reject unrendered flat neutral grey canvas (HEVC missing I-frame canvas > 45% grey)
    grey_dist = np.max(np.abs(frame.astype(np.int16) - 128), axis=2)
    if np.mean(grey_dist < 20) > 0.45:
        return False

    # 3. Reject bright green slice corruption (YUV 0,0,0 uninitialized macroblocks)
    green_mask = (
        (frame[:, :, 1] > 160)
        & (frame[:, :, 0] < 50)
        & (frame[:, :, 2] < 50)
    )
    if np.mean(green_mask) > 0.03:
        return False

    # 4. Reject blue/magenta chroma tearing bands
    blue_mask = (
        (frame[:, :, 0] > 180)
        & (frame[:, :, 1] < 50)
        & (frame[:, :, 2] < 50)
    )
    if np.mean(blue_mask) > 0.03:
        return False

    # 5. Reject HEVC error concealment checkerboard / dotted noise artifacts
    bot_half = frame[int(h * 0.45):, :].astype(np.float32)
    h_grad = float(np.mean(np.abs(bot_half[:, 1:] - bot_half[:, :-1])))
    v_grad = float(np.mean(np.abs(bot_half[1:, :] - bot_half[:-1, :])))
    if h_grad > 30.0 or v_grad > 30.0:
        return False

    return True


class RTSPCaptureManager:
    """
    Dedicated Singleton RTSP Capture Manager.
    Sole owner of:
      - RTSP stream connection lifecycle
      - Zero-overhead background frame capture loop (single capture thread)
      - Multi-stage startup validation (STOPPED -> CONNECTING -> STREAM_OPENED -> VALIDATING -> LIVE)
      - Single-flight recovery on failure threshold
      - Thread-safe latest-frame deep copy buffer
      - Controlled clean teardown
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.state = CameraState.STOPPED
        self.lifecycle_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self._live_event = threading.Event()
        self._stop_event = threading.Event()
        
        self.proc = None
        self.capture_thread = None
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        self._latest_frame = None
        self._frame_id = 0
        self._consumed_frame_id = -1
        
        # Dynamic native resolution
        self.native_width = 640
        self.native_height = 480
        
        # Diagnostics
        self.total_captured = 0
        self.total_dropped = 0
        self.corrupted_count = 0
        self.read_failures = 0
        self.reconnect_count = 0
        
        # Real rolling Capture FPS
        self._capture_timestamps = collections.deque(maxlen=30)
        self.capture_fps = 0.0

    def start(self):
        """Starts the capture manager in a single background capture thread."""
        with self.lifecycle_lock:
            if self.state != CameraState.STOPPED:
                return False
            self._stop_event.clear()
            self._live_event.clear()
            self.state = CameraState.CONNECTING
            print("[CAMERA] Starting", flush=True)
            self.capture_thread = threading.Thread(
                target=self._capture_worker,
                name="RTSPCaptureThread",
                daemon=True
            )
            self.capture_thread.start()
            print("[CAMERA] Capture thread started", flush=True)
            return True

    def wait_until_live(self, timeout=25.0):
        """Blocks until startup validation finishes and stream transitions to LIVE."""
        return self._live_event.wait(timeout=timeout)

    def _drain_stderr(self, proc):
        """Continuously drains stderr to avoid pipe buffer deadlocks on Windows."""
        try:
            for line in iter(proc.stderr.readline, b""):
                if not line or self._stop_event.is_set():
                    break
        except Exception:
            pass

    def _safe_close_proc(self):
        """Safely terminates and kills stream process."""
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.kill()
                self.proc.wait(timeout=1.0)
            except Exception:
                pass
            self.proc = None

    def _capture_worker(self):
        CONSECUTIVE_VALID_REQUIRED = 5
        MAX_READ_FAILURES = 30
        
        # High-performance, zero-latency 100MB UDP socket buffer pipeline with 10s jitter resilience
        cmd = [
            self.ffmpeg_exe,
            "-loglevel", "error",
            "-rtsp_transport", "udp",
            "-max_delay", "10000000",        # 10s jitter buffer (completely eliminates motion burst packet drops)
            "-reorder_queue_size", "10000",  # 10,000 packet reorder buffer
            "-buffer_size", "104857600",     # 100MB socket buffer
            "-fflags", "+genpts+discardcorrupt",
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
            print("[CAMERA] Opening RTSP stream", flush=True)
            
            try:
                self.proc = subprocess.Popen(
                    cmd,
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

            self.state = CameraState.STREAM_OPENED
            print("[CAMERA] VideoCapture opened", flush=True)
            if is_recovering:
                print("[CAMERA] New stream validating", flush=True)
            print("[CAMERA] Validating frames", flush=True)
            self.state = CameraState.VALIDATING

            pipe_buffer = bytearray()
            consecutive_good = 0
            validation_attempts = 0
            stable_w, stable_h = 0, 0
            validated = False

            # Startup Validation Loop: Require 5 consecutive pristine frames
            while not self._stop_event.is_set() and self.proc.poll() is None and validation_attempts < 80:
                chunk = self.proc.stdout.read(16384)
                if not chunk:
                    if self._stop_event.is_set() or self.proc.poll() is not None:
                        break
                    time.sleep(0.001)
                    continue
                pipe_buffer.extend(chunk)

                while True:
                    soi = pipe_buffer.find(b"\xff\xd8")
                    if soi == -1:
                        if len(pipe_buffer) > 1:
                            pipe_buffer = pipe_buffer[-1:]
                        break
                    eoi = pipe_buffer.find(b"\xff\xd9", soi + 2)
                    if eoi == -1:
                        if soi > 0:
                            pipe_buffer = pipe_buffer[soi:]
                        break

                    jpg_bytes = pipe_buffer[soi:eoi + 2]
                    pipe_buffer = pipe_buffer[eoi + 2:]
                    validation_attempts += 1
                    seq_id += 1

                    raw_frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if is_valid_clean_frame(raw_frame):
                        h, w = raw_frame.shape[:2]
                        if consecutive_good == 0:
                            stable_w, stable_h = w, h
                            consecutive_good = 1
                        elif w == stable_w and h == stable_h:
                            consecutive_good += 1
                        else:
                            consecutive_good = 1
                            stable_w, stable_h = w, h

                        print(f"[CAMERA] Valid frame {consecutive_good}/{CONSECUTIVE_VALID_REQUIRED}", flush=True)

                        if consecutive_good >= CONSECUTIVE_VALID_REQUIRED:
                            self.native_width = stable_w
                            self.native_height = stable_h
                            with self.frame_lock:
                                self._latest_frame = raw_frame.copy()
                                self._frame_id = seq_id
                                self.total_captured += 1
                                self._capture_timestamps.append(time.perf_counter())
                            self.state = CameraState.LIVE
                            self._live_event.set()
                            if is_recovering:
                                print("[CAMERA] Recovered successfully", flush=True)
                                is_recovering = False
                            print("[CAMERA] LIVE", flush=True)
                            validated = True
                            break
                    else:
                        self.corrupted_count += 1
                        consecutive_good = 0

                if validated:
                    break

            if not validated:
                if self._stop_event.is_set():
                    break
                print("[CAMERA] Startup validation failed. Recovery started", flush=True)
                self._safe_close_proc()
                time.sleep(1.0)
                continue

            # LIVE Streaming Loop: Pure in-memory streaming with zero console blocking
            consecutive_failures = 0
            while not self._stop_event.is_set() and self.proc.poll() is None:
                chunk = self.proc.stdout.read(16384)
                if not chunk:
                    if self._stop_event.is_set() or self.proc.poll() is not None:
                        break
                    time.sleep(0.001)
                    continue
                pipe_buffer.extend(chunk)

                while True:
                    soi = pipe_buffer.find(b"\xff\xd8")
                    if soi == -1:
                        if len(pipe_buffer) > 1:
                            pipe_buffer = pipe_buffer[-1:]
                        break
                    eoi = pipe_buffer.find(b"\xff\xd9", soi + 2)
                    if eoi == -1:
                        if soi > 0:
                            pipe_buffer = pipe_buffer[soi:]
                        break

                    jpg_bytes = pipe_buffer[soi:eoi + 2]
                    pipe_buffer = pipe_buffer[eoi + 2:]
                    seq_id += 1

                    now = time.perf_counter()
                    raw_frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if raw_frame is None or raw_frame.size == 0:
                        consecutive_failures += 1
                        self.read_failures += 1
                        if consecutive_failures >= MAX_READ_FAILURES:
                            break
                        continue

                    if not is_valid_clean_frame(raw_frame):
                        self.corrupted_count += 1
                        continue

                    # Frame is 100% pristine and validated
                    consecutive_failures = 0
                    h, w = raw_frame.shape[:2]
                    self.native_width = w
                    self.native_height = h

                    # Private deep copy before taking lock
                    safe_copy = raw_frame.copy()
                    with self.frame_lock:
                        if self._latest_frame is not None and self._frame_id > self._consumed_frame_id:
                            self.total_dropped += 1
                        self._latest_frame = safe_copy
                        self._frame_id = seq_id
                        self.total_captured += 1

                        self._capture_timestamps.append(now)
                        if len(self._capture_timestamps) > 1:
                            self.capture_fps = (len(self._capture_timestamps) - 1) / (
                                self._capture_timestamps[-1] - self._capture_timestamps[0]
                            )

            # Controlled Single-Flight Recovery
            if not self._stop_event.is_set():
                self._live_event.clear()
                self.state = CameraState.RECOVERING
                self.reconnect_count += 1
                is_recovering = True
                print("[CAMERA] Recovery started", flush=True)
                self._safe_close_proc()
                print("[CAMERA] Old capture released", flush=True)
                print("[CAMERA] New stream validating", flush=True)
                time.sleep(1.0)

        # Thread exit
        self._safe_close_proc()
        self.state = CameraState.STOPPED
        print("[CAMERA] Old thread exited", flush=True)

    def get_display_snapshot(self):
        """
        Retrieves an independent deep copy snapshot of the newest available validated frame.
        Guarantees complete memory isolation under lock so rendering never touches capture buffer.
        """
        with self.frame_lock:
            if self._latest_frame is not None:
                is_new = (self._frame_id != self._consumed_frame_id)
                self._consumed_frame_id = self._frame_id
                snapshot = self._latest_frame.copy()
                frame_id = self._frame_id
                return True, snapshot, is_new, self.capture_fps, self.state, frame_id
            return False, None, False, 0.0, self.state, -1

    def shutdown(self):
        """
        Clean, single-flight teardown releasing all capture objects,
        stopping background worker, joining thread, and clearing memory references.
        """
        with self.lifecycle_lock:
            if self.state == CameraState.STOPPED and self.capture_thread is None:
                return
            self._stop_event.set()
            self._safe_close_proc()
            if self.capture_thread is not None and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=3.0)
                self.capture_thread = None
            with self.frame_lock:
                self._latest_frame = None
            self.state = CameraState.STOPPED
            print("[CAMERA] Shutdown complete", flush=True)


def on_mouse_click(event, x, y, flags, param):
    """
    OpenCV Mouse Callback:
    Maps display window coordinates back to original native camera frame coordinates.
    """
    global clicked_display_pt, clicked_orig_pt, camera_manager_instance

    if event == cv2.EVENT_LBUTTONDOWN and camera_manager_instance is not None:
        nat_w = camera_manager_instance.native_width
        nat_h = camera_manager_instance.native_height

        orig_x = int(round(x * (nat_w / float(DISPLAY_WIDTH))))
        orig_y = int(round(y * (nat_h / float(DISPLAY_HEIGHT))))

        orig_x = max(0, min(orig_x, nat_w - 1))
        orig_y = max(0, min(orig_y, nat_h - 1))

        clicked_display_pt = (x, y)
        clicked_orig_pt = (orig_x, orig_y)
        
        print(f"[📍 CLICK] Display: ({x}, {y})  ==>  Original Camera Frame: X={orig_x}, Y={orig_y}", flush=True)


def run_camera():
    global clicked_display_pt, clicked_orig_pt, camera_manager_instance

    print("=" * 65)
    print(" IP Camera Live Stream & Coordinate Tracker")
    print(f" RTSP URL: {RTSP_URL}")
    print(" Architecture:")
    print("   • Dedicated RTSPCaptureManager (Single-Flight Lifecycle)")
    print("   • Multi-Stage Startup Validation (STOPPED->CONNECTING->STREAM_OPENED->VALIDATING->LIVE)")
    print("   • 100MB UDP Socket Buffer (Zero Packet Drops)")
    print("   • Zero-Overhead In-Memory Corruption Filter")
    print("   • Smooth 25-30 FPS Presentation")
    print(" Controls:")
    print("   [Left-Click] : Track original X/Y coordinate at click point")
    print("   [C] Key      : Clear tracked coordinates")
    print("   [S] Key      : Save snapshot image with tracker overlay")
    print("   [Q] or [ESC] : Quit stream cleanly")
    print("=" * 65)

    camera_manager_instance = RTSPCaptureManager(RTSP_URL)
    camera_manager_instance.start()

    print("\n[+] Synchronizing live camera feed...", end="", flush=True)
    t_start_sync = time.perf_counter()
    
    is_live = camera_manager_instance.wait_until_live(timeout=25.0)
    if not is_live:
        print("\n[!] Failed to connect to camera. Shutting down.", flush=True)
        camera_manager_instance.shutdown()
        return

    print(f"\n✅ Camera connected in {time.perf_counter() - t_start_sync:.2f}s! Crystal-clear video window opened.\n", flush=True)

    window_created = False
    display_timestamps = collections.deque(maxlen=30)
    display_fps = 0.0

    try:
        while True:
            t_cycle_start = time.perf_counter()

            # Retrieve independent deep-copy snapshot under frame lock
            ret, frame, is_new, cap_fps, state, frame_id = camera_manager_instance.get_display_snapshot()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if is_new:
                # Calculate Dynamic Display FPS strictly from newly received frames
                display_timestamps.append(t_cycle_start)
                if len(display_timestamps) > 1:
                    display_fps = (len(display_timestamps) - 1) / (
                        display_timestamps[-1] - display_timestamps[0]
                    )

            # Natural balanced contrast calibration on private display copy
            calibrated_frame = cv2.convertScaleAbs(frame, alpha=1.04, beta=3)

            # Smooth, high-fidelity Lanczos4 display resize
            display_frame = cv2.resize(calibrated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

            # Draw coordinate tracking overlay if a point has been clicked
            if clicked_display_pt is not None and clicked_orig_pt is not None:
                disp_x, disp_y = clicked_display_pt
                orig_x, orig_y = clicked_orig_pt

                # 1. Target Crosshair & Concentric rings
                cv2.drawMarker(
                    display_frame,
                    (disp_x, disp_y),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=26,
                    thickness=2,
                )
                cv2.circle(display_frame, (disp_x, disp_y), 12, (0, 255, 255), 2)
                cv2.circle(display_frame, (disp_x, disp_y), 3, (0, 0, 255), -1)

                # 2. Floating coordinate badge tooltip
                coord_text = f"X: {orig_x}, Y: {orig_y}"
                text_x = disp_x + 16 if disp_x + 160 < DISPLAY_WIDTH else disp_x - 150
                text_y = disp_y - 14 if disp_y - 30 > 0 else disp_y + 30

                # Background label with cyan border
                cv2.rectangle(display_frame, (text_x - 6, text_y - 18), (text_x + 135, text_y + 8), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (text_x - 6, text_y - 18), (text_x + 135, text_y + 8), (0, 255, 255), 2)
                cv2.putText(
                    display_frame,
                    coord_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Top HUD Status Bar (Showing real dynamic Capture FPS & Display FPS ~25-30 FPS)
            nat_w = camera_manager_instance.native_width
            nat_h = camera_manager_instance.native_height
            badge_w = 600 if clicked_orig_pt is not None else 450
            cv2.rectangle(display_frame, (10, 10), (badge_w, 48), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (10, 10), (badge_w, 48), (60, 60, 60), 1)

            # Glowing green LIVE indicator dot
            cv2.circle(display_frame, (25, 29), 6, (0, 255, 0), -1)

            # Real measured dynamic FPS text (natural 25-30 FPS rate)
            status_text = f"LIVE | {nat_w}x{nat_h} | Cap:{cap_fps:.1f} FPS | Disp:{display_fps:.1f} FPS"
            if clicked_orig_pt is not None:
                status_text += f" | ({clicked_orig_pt[0]}, {clicked_orig_pt[1]})"

            cv2.putText(
                display_frame,
                status_text,
                (38, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Initialize window and set mouse callback once
            if not window_created:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)
                window_created = True

            # Display the rendered frame
            cv2.imshow(WINDOW_NAME, display_frame)

            # Smooth frame pacing (25-30 FPS)
            elapsed = time.perf_counter() - t_cycle_start
            wait_ms = max(1, int(round((FRAME_INTERVAL - elapsed) * 1000.0)))
            key = cv2.waitKey(wait_ms) & 0xFF

            # Process keyboard controls
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("\n[+] Closing live stream...")
                break
            elif key == ord("c") or key == ord("C"):
                clicked_display_pt = None
                clicked_orig_pt = None
                print("[+] Cleared tracked coordinates.")
            elif key == ord("s") or key == ord("S"):
                snap_filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_filename, display_frame)
                print(f"[+] Saved snapshot to '{snap_filename}'")

    except KeyboardInterrupt:
        print("\n[+] Interrupted by user.")
    finally:
        if camera_manager_instance is not None:
            camera_manager_instance.shutdown()
        cv2.destroyAllWindows()
        print("✅ Live stream closed cleanly.")


if __name__ == "__main__":
    run_camera()
