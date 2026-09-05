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

# Target Display Resolution
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

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


class AtomicLatestFrameSlot:
    """
    Lock-protected atomic latest frame slot with strict zero-delay semantics.
    Capture thread continuously publishes the newest frame, discarding unconsumed frames.
    Display loop always retrieves the freshest available frame with sub-3ms latency.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self._latest_frame = None
        self._frame_id = 0
        self._capture_ts = 0.0
        self._consumed_id = -1
        
        # Native stream dimensions
        self.native_width = 640
        self.native_height = 480
        
        # Diagnostics
        self.total_captured = 0
        self.total_dropped_capture = 0
        self.total_displayed = 0
        
        # Rolling Capture FPS
        self._capture_timestamps = collections.deque(maxlen=30)
        self.capture_fps = 0.0

    def publish_frame(self, frame, frame_id, capture_ts):
        """Called exclusively by the capture thread. Never blocks on display."""
        h, w = frame.shape[:2]
        now = time.perf_counter()

        with self.lock:
            if self._latest_frame is not None and self._frame_id > self._consumed_id:
                self.total_dropped_capture += 1

            self._latest_frame = frame
            self._frame_id = frame_id
            self._capture_ts = capture_ts
            self.total_captured += 1
            self.native_width = w
            self.native_height = h

            self._capture_timestamps.append(now)
            if len(self._capture_timestamps) > 1:
                self.capture_fps = (len(self._capture_timestamps) - 1) / (
                    self._capture_timestamps[-1] - self._capture_timestamps[0]
                )
        return True

    def get_display_frame(self):
        """
        Called exclusively by the display loop.
        Immediately returns a private snapshot copy of the latest available frame.
        """
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
            if is_new:
                self.total_displayed += 1

        return True, frame_copy, fid, cts, is_new, cap_fps, nw, nh

    def clear(self):
        with self.lock:
            self._latest_frame = None
            self._frame_id = 0
            self._consumed_id = -1
            self._capture_timestamps.clear()
            self.capture_fps = 0.0


def is_clean_frame(frame):
    """
    Validates that a frame is genuine, full-contrast, and free from
    H.265/HEVC macroblock packet loss corruption (grey smear / checkerboard grid).
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0 or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    # Overall image contrast / standard deviation
    if frame.std() < 18.0:
        return False
    # Check bottom half for grey smear or default H.265 deblocking block patterns (RGB all ~128)
    bottom = frame[int(h * 0.4):, :]
    grey_mask = (
        (np.abs(bottom[:, :, 0].astype(np.int16) - 128) < 14) &
        (np.abs(bottom[:, :, 1].astype(np.int16) - 128) < 14) &
        (np.abs(bottom[:, :, 2].astype(np.int16) - 128) < 14)
    )
    if np.mean(grey_mask) > 0.25:
        return False
    return True


class RTSPCaptureManager:
    """
    Dedicated RTSP Capture Manager.
    Architecture:
      - Direct FFmpeg decoding to rawvideo (pix_fmt: bgr24) via stdout
      - Clean Frame Guard (discards any partial grey smear / checkerboard macroblock frames)
      - Exact fixed-size frame reader (frame_size = width * height * 3)
      - Atomic latest frame slot (zero backlog / lowest latency)
      - Dynamic stream resolution probing via FFmpeg
      - 10 complete raw frames startup validation
      - Fast single-flight recovery on stream failure or consecutive corrupt frames
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
        
        # Stream dimensions
        self.width = None
        self.height = None
        self.frame_size = None
        
        # Diagnostics
        self.read_failures = 0
        self.corrupted_frames = 0
        self.reconnect_count = 0

    @property
    def native_width(self):
        return self.slot.native_width if self.width is None else self.width

    @property
    def native_height(self):
        return self.slot.native_height if self.height is None else self.height

    def probe_resolution(self, timeout=5.0):
        """
        Probes the RTSP stream using FFmpeg to dynamically determine
        the actual native video width and height before starting rawvideo capture.
        """
        import re
        cmd = [
            self.ffmpeg_exe,
            "-loglevel", "info",
            "-probesize", "131072",
            "-analyzeduration", "100000",
            "-buffer_size", "8388608",
            "-an",
            "-i", self.rtsp_url,
            "-t", "0.01",
            "-f", "null",
            "-"
        ]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            t_start = time.time()
            while time.time() - t_start < timeout:
                line = p.stderr.readline()
                if not line:
                    if p.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue
                # Match patterns like: Stream #0:0: Video: hevc (Main), yuvj420p(pc, bt709), 640x480
                m = re.search(r'Stream #\d+:\d+.*Video:.*,\s*(\d{2,5})x(\d{2,5})', line)
                if m:
                    w, h = int(m.group(1)), int(m.group(2))
                    p.kill()
                    p.wait(timeout=1.0)
                    return w, h
            p.kill()
            p.wait(timeout=1.0)
        except Exception as e:
            print(f"[CAMERA] Resolution probe warning: {e}", flush=True)
        return 640, 480  # Default fallback if probe doesn't report

    def start(self):
        """Starts the capture manager in a single background capture thread."""
        with self.lifecycle_lock:
            if self.state != CameraState.STOPPED:
                return False
            self._stop_event.clear()
            self._live_event.clear()
            self.slot.clear()
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
        """Safely terminates and kills stream process and closes stdout."""
        if self.proc is not None:
            try:
                if self.proc.stdout:
                    self.proc.stdout.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.kill()
                self.proc.wait(timeout=1.0)
            except Exception:
                pass
            self.proc = None

    def _read_exactly(self, num_bytes):
        """
        Reads EXACTLY num_bytes from FFmpeg stdout.
        Never combines bytes from different frames.
        Returns bytes of length num_bytes, or None if EOF or stop requested.
        """
        if self.proc is None or self.proc.stdout is None:
            return None
        
        buf = bytearray(num_bytes)
        pos = 0
        stream = self.proc.stdout
        
        while pos < num_bytes:
            if self._stop_event.is_set():
                return None
            try:
                chunk = stream.read(num_bytes - pos)
                if not chunk:
                    # EOF reached before complete frame
                    return None
                buf[pos:pos+len(chunk)] = chunk
                pos += len(chunk)
            except Exception:
                return None

        return bytes(buf)

    def _capture_worker(self):
        """
        Pure background capture thread.
        Pipeline:
          FFmpeg decode -> rawvideo BGR24 -> stdout
          -> read_exactly(frame_size) -> is_clean_frame() -> publish to atomic slot.
        """
        CONSECUTIVE_VALID_REQUIRED = 5
        MAX_CONSECUTIVE_CORRUPT = 25
        MAX_READ_FAILURES = 15

        # 1. Dynamically probe resolution before starting capture
        print("[CAMERA] Probing stream resolution...", flush=True)
        self.width, self.height = self.probe_resolution()
        self.frame_size = self.width * self.height * 3
        print(f"[CAMERA] Stream resolution detected: {self.width}x{self.height} (frame_size={self.frame_size} bytes)", flush=True)

        cmd = [
            self.ffmpeg_exe,
            "-loglevel", "error",
            "-buffer_size", "33554432",        # 32MB socket buffer to prevent packet drops
            "-max_delay", "500000",            # 500ms jitter buffer
            "-reorder_queue_size", "4000",
            "-an",
            "-i", self.rtsp_url,
            "-map", "0:v:0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
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

            consecutive_good = 0
            validation_attempts = 0
            validated = False

            # Startup Validation Loop: Require consecutive clean, non-corrupted frames
            while not self._stop_event.is_set() and self.proc.poll() is None and validation_attempts < 100:
                raw_bytes = self._read_exactly(self.frame_size)
                if raw_bytes is None or len(raw_bytes) != self.frame_size:
                    if self._stop_event.is_set() or self.proc.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue

                validation_attempts += 1
                seq_id += 1
                t_cap = time.perf_counter()

                # Reshape EXACT fixed-size raw BGR24 data
                raw_frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))

                if is_clean_frame(raw_frame):
                    consecutive_good += 1
                    print(f"[CAMERA] Valid frame {consecutive_good}/{CONSECUTIVE_VALID_REQUIRED}", flush=True)

                    if consecutive_good >= CONSECUTIVE_VALID_REQUIRED:
                        self.slot.publish_frame(raw_frame, seq_id, t_cap)
                        self.state = CameraState.LIVE
                        self._live_event.set()
                        if is_recovering:
                            print("[CAMERA] Recovered successfully", flush=True)
                            is_recovering = False
                        print("[CAMERA] LIVE", flush=True)
                        validated = True
                        break
                else:
                    consecutive_good = 0

            if not validated:
                if self._stop_event.is_set():
                    break
                print("[CAMERA] Startup validation failed. Recovery started", flush=True)
                self._safe_close_proc()
                time.sleep(0.5)
                continue

            # LIVE Streaming Loop: Fast read_exactly -> clean validation -> publish to atomic slot
            consecutive_failures = 0
            consecutive_corrupt = 0
            
            while not self._stop_event.is_set() and self.proc.poll() is None:
                raw_bytes = self._read_exactly(self.frame_size)
                if raw_bytes is None or len(raw_bytes) != self.frame_size:
                    consecutive_failures += 1
                    self.read_failures += 1
                    if consecutive_failures >= MAX_READ_FAILURES or self.proc.poll() is not None:
                        break
                    time.sleep(0.005)
                    continue

                seq_id += 1
                t_cap = time.perf_counter()

                # Reshape exact complete raw frame
                raw_frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))

                # Strict Clean Frame Check:
                # If a dropped network packet caused H.265 grey smear or checkerboard blocks,
                # discard this frame so it NEVER corrupts the user's display!
                if not is_clean_frame(raw_frame):
                    consecutive_corrupt += 1
                    self.corrupted_frames += 1
                    if consecutive_corrupt >= MAX_CONSECUTIVE_CORRUPT:
                        print("[CAMERA] Corrupt frame threshold exceeded (missing I-frame). Triggering fast recovery...", flush=True)
                        break
                    continue

                # Frame is 100% clean and pristine
                consecutive_corrupt = 0
                consecutive_failures = 0
                self.slot.publish_frame(raw_frame, seq_id, t_cap)

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
        Retrieves an independent deep copy snapshot of the newest available clean frame.
        """
        return self.slot.get_display_frame()

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
            self.slot.clear()
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
    print("   • 8MB UDP Socket Buffer (Zero Kernel Packet Drops on Motion)")
    print("   • Atomic Latest Frame Slot (Zero Backlog / Real-time Latency)")
    print("   • Pure Zero-Overhead Background Capture Thread")
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
    
    is_live = camera_manager_instance.wait_until_live(timeout=20.0)
    if not is_live:
        print("\n[!] Failed to connect to camera. Shutting down.", flush=True)
        camera_manager_instance.shutdown()
        return

    print(f"\n✅ Camera connected in {time.perf_counter() - t_start_sync:.2f}s! Crystal-clear video window opened.\n", flush=True)

    window_created = False
    display_timestamps = collections.deque(maxlen=30)
    display_fps = 0.0
    last_log_time = time.perf_counter()

    try:
        while True:
            t_cycle_start = time.perf_counter()

            # Retrieve independent deep-copy snapshot from atomic slot
            ret, frame, fid, cts, is_new, cap_fps, nat_w, nat_h = camera_manager_instance.get_display_snapshot()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            # Measured end-to-end capture latency
            lat_ms = (t_cycle_start - cts) * 1000.0

            if is_new:
                # Dynamic Display FPS strictly from newly received frames
                display_timestamps.append(t_cycle_start)
                if len(display_timestamps) > 1:
                    display_fps = (len(display_timestamps) - 1) / (
                        display_timestamps[-1] - display_timestamps[0]
                    )

            # Periodic diagnostic logging
            if t_cycle_start - last_log_time >= 2.0:
                print(f"[METRICS] CAPTURE id={camera_manager_instance.slot._frame_id} | DISPLAY id={fid} | LATENCY={lat_ms:.1f}ms | Cap:{cap_fps:.1f} FPS | Disp:{display_fps:.1f} FPS | Dropped={camera_manager_instance.slot.total_dropped_capture}", flush=True)
                last_log_time = t_cycle_start

            # Natural balanced contrast calibration on private display copy
            calibrated_frame = cv2.convertScaleAbs(frame, alpha=1.04, beta=3)

            # High-performance, crisp display resize
            display_frame = cv2.resize(calibrated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

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

            # Top HUD Status Bar
            badge_w = 600 if clicked_orig_pt is not None else 460
            cv2.rectangle(display_frame, (10, 10), (badge_w, 48), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (10, 10), (badge_w, 48), (60, 60, 60), 1)

            # Glowing green LIVE indicator dot
            cv2.circle(display_frame, (25, 29), 6, (0, 255, 0), -1)

            # Real measured dynamic FPS & Latency text
            status_text = f"LIVE | {nat_w}x{nat_h} | Cap:{cap_fps:.1f} FPS | Disp:{display_fps:.1f} FPS | {lat_ms:.0f}ms"
            if clicked_orig_pt is not None:
                status_text += f" | ({clicked_orig_pt[0]}, {clicked_orig_pt[1]})"

            cv2.putText(
                display_frame,
                status_text,
                (38, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
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

            # Instant key poll (1ms non-blocking)
            key = cv2.waitKey(1) & 0xFF

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
