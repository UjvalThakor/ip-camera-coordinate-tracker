import os
import sys
import time
import cv2
import numpy as np

# Force reliable TCP transport for RTSP stream stability
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Camera RTSP Stream URL for IP 10.27.1.77
RTSP_URL = "rtsp://admin:admin123@10.27.1.77:554/Streaming/Channels/102/"

# Display window dimensions
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Global coordinate tracking state
latest_orig_dim = (640, 480) # (orig_w, orig_h)
clicked_display_pt = None    # (disp_x, disp_y)
clicked_orig_pt = None       # (orig_x, orig_y)


def on_mouse_click(event, x, y, flags, param):
    """
    OpenCV Mouse Callback:
    Captures mouse left-click on the live video window and maps
    display coordinates to original camera-frame coordinates.
    """
    global clicked_display_pt, clicked_orig_pt, latest_orig_dim

    if event == cv2.EVENT_LBUTTONDOWN:
        orig_w, orig_h = latest_orig_dim
        
        # Calculate original camera frame coordinates
        orig_x = int(round(x * (orig_w / float(DISPLAY_WIDTH))))
        orig_y = int(round(y * (orig_h / float(DISPLAY_HEIGHT))))

        # Clamp to valid image bounds
        orig_x = max(0, min(orig_x, orig_w - 1))
        orig_y = max(0, min(orig_y, orig_h - 1))

        clicked_display_pt = (x, y)
        clicked_orig_pt = (orig_x, orig_y)
        
        print(f"[📍 CLICK] Window Display: ({x}, {y}) -> Original Camera Frame: X={orig_x}, Y={orig_y}")


def run_camera():
    global latest_orig_dim, clicked_display_pt, clicked_orig_pt

    print("=" * 65)
    print("IP Camera Live Stream & Coordinate Tracker")
    print(f"Connecting to RTSP Camera: {RTSP_URL}")
    print("Controls:")
    print("  -> Left-Click anywhere on the video to track original X/Y coordinates")
    print("  -> Press 'c' to clear tracked coordinates")
    print("  -> Press 'q' or ESC on the video window to quit")
    print("=" * 65)

    window_name = "IP Camera Live Stream (10.27.1.77)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse_click)

    # 1. Single persistent VideoCapture connection
    print("\n[+] Initializing camera connection. Please wait...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Could not connect to camera RTSP stream.")
        print("Please verify the camera IP (10.27.1.77) and network connection.")
        return

    print("✅ Camera connected! Streaming live video...")

    # Real FPS measurement variables
    fps_start_time = time.time()
    frame_count = 0
    actual_fps = 0.0

    # Main continuous live streaming loop
    while True:
        # Read next incoming frame
        ret, frame = cap.read()

        # Handle dropped/invalid frames safely
        if not ret or frame is None:
            time.sleep(0.001)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            continue

        # Dynamic original camera resolution detection
        orig_h, orig_w = frame.shape[:2]
        latest_orig_dim = (orig_w, orig_h)

        # Compute real live FPS from successfully received frames
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            actual_fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        # Resize to fixed display window for consistent rendering & mouse mapping
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # If user clicked, draw marker and coordinates directly at the click position
        if clicked_display_pt is not None and clicked_orig_pt is not None:
            disp_x, disp_y = clicked_display_pt
            orig_x, orig_y = clicked_orig_pt

            # 1. Draw crosshair marker at the exact clicked display pixel
            cv2.drawMarker(
                display_frame,
                (disp_x, disp_y),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=22,
                thickness=2,
            )
            cv2.circle(display_frame, (disp_x, disp_y), 9, (0, 255, 255), 2)
            cv2.circle(display_frame, (disp_x, disp_y), 2, (0, 0, 255), -1)

            # 2. Draw floating coordinate tag beside the marker
            coord_text = f"X: {orig_x}, Y: {orig_y}"
            
            # Position label to stay inside window bounds
            text_x = disp_x + 14 if disp_x + 150 < DISPLAY_WIDTH else disp_x - 140
            text_y = disp_y - 12 if disp_y - 25 > 0 else disp_y + 25

            # Background label box for high contrast readability
            cv2.rectangle(display_frame, (text_x - 4, text_y - 16), (text_x + 125, text_y + 6), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (text_x - 4, text_y - 16), (text_x + 125, text_y + 6), (0, 255, 255), 1)
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

        # Render top status bar (Resolution, FPS, and Selected Coordinate)
        badge_w = 490 if clicked_orig_pt is not None else 350
        cv2.rectangle(display_frame, (10, 10), (badge_w, 48), (0, 0, 0), -1)
        
        # Green LIVE indicator dot
        cv2.circle(display_frame, (25, 29), 6, (0, 255, 0), -1)

        # Status text
        status_text = f"LIVE | {orig_w}x{orig_h} | {actual_fps:.1f} FPS"
        if clicked_orig_pt is not None:
            status_text += f" | Point: ({clicked_orig_pt[0]}, {clicked_orig_pt[1]})"

        cv2.putText(
            display_frame,
            status_text,
            (38, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Immediately display current live frame
        cv2.imshow(window_name, display_frame)

        # Continuous waitKey to keep window alive and process keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("\n[+] Exiting live stream...")
            break
        elif key == ord("c") or key == ord("C"):
            clicked_display_pt = None
            clicked_orig_pt = None
            print("[+] Cleared tracked coordinates.")

    # Clean release of the persistent camera stream
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Stream closed cleanly.")


if __name__ == "__main__":
    run_camera()