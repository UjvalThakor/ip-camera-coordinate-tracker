import os
import sys
import time
import cv2
import numpy as np

# Force reliable TCP transport for RTSP stream stability
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Camera RTSP Stream URL for IP 10.27.1.77
RTSP_URL = "rtsp://admin:admin123@10.27.1.77:554/Streaming/Channels/102/"

# Global coordinate tracking state
clicked_point_frame = None   # Coordinates in original camera frame (orig_x, orig_y)
clicked_point_display = None # Coordinates in display window (disp_x, disp_y)
display_scale_factors = (1.0, 1.0) # (scale_x, scale_y)


def mouse_callback(event, x, y, flags, param):
    """
    Handle mouse click events and accurately map display coordinates
    back to the original camera frame resolution.
    """
    global clicked_point_frame, clicked_point_display, display_scale_factors

    if event == cv2.EVENT_LBUTTONDOWN:
        scale_x, scale_y = display_scale_factors
        # Map display window coordinate back to original camera coordinate
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)

        clicked_point_display = (x, y)
        clicked_point_frame = (orig_x, orig_y)
        print(f"[📍 CLICK DETECTED] Display: ({x}, {y}) -> Original Frame: (X={orig_x}, Y={orig_y})")


def run_camera():
    global display_scale_factors, clicked_point_frame, clicked_point_display

    print("=" * 65)
    print("IP Camera Live Stream & Coordinate Tracker")
    print(f"Connecting to RTSP Camera: {RTSP_URL}")
    print("Controls:")
    print("  -> Click anywhere on the video to track original X/Y coordinates")
    print("  -> Press 'c' to clear the tracked point")
    print("  -> Press 'q' or ESC on the video window to quit")
    print("=" * 65)

    window_name = "IP Camera Live Stream (10.27.1.77)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.setMouseCallback(window_name, mouse_callback)

    # 1. Single persistent VideoCapture connection
    print("\n[+] Initializing camera connection. Please wait...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Could not connect to camera RTSP stream.")
        print("Please verify the camera IP (10.27.1.77) and network connection.")
        return

    print("✅ Camera connection established! Starting live stream loop...")

    # Real FPS measurement variables
    fps_start_time = time.time()
    frame_count = 0
    actual_fps = 0.0

    # Main continuous frame-reading and display loop
    while True:
        # Read next incoming frame
        ret, frame = cap.read()

        # Handle failed/invalid/corrupted frame reads gracefully
        if not ret or frame is None:
            # Yield CPU briefly and continue reading next frame
            time.sleep(0.001)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            continue

        # Dynamic resolution detection (not hardcoded)
        orig_h, orig_w = frame.shape[:2]

        # Calculate actual live FPS from successfully received frames
        frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            actual_fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        # Prepare display copy so we don't modify the raw data
        display_frame = frame.copy()

        # Get current display window size to compute accurate mouse coordinate scaling
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w > 0 and win_h > 0:
                display_scale_factors = (orig_w / float(win_w), orig_h / float(win_h))
            else:
                display_scale_factors = (1.0, 1.0)
        except Exception:
            display_scale_factors = (1.0, 1.0)

        # Draw clicked point / crosshair on the original frame
        if clicked_point_frame is not None:
            pt_x, pt_y = clicked_point_frame
            # Draw crosshair target marker
            cv2.drawMarker(
                display_frame,
                (pt_x, pt_y),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            cv2.circle(display_frame, (pt_x, pt_y), 8, (0, 255, 255), 2)
            
            # Draw coordinate text near the point
            coord_label = f"({pt_x}, {pt_y})"
            cv2.putText(
                display_frame,
                coord_label,
                (pt_x + 12, pt_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Render top status badge (Resolution, FPS, and Coordinates)
        badge_w = 480 if clicked_point_frame is not None else 360
        cv2.rectangle(display_frame, (10, 10), (badge_w, 50), (0, 0, 0), -1)
        
        # Green LIVE indicator dot
        cv2.circle(display_frame, (26, 30), 6, (0, 255, 0), -1)

        # Status text with dynamically detected resolution and measured FPS
        status_text = f"LIVE | {orig_w}x{orig_h} | {actual_fps:.1f} FPS"
        if clicked_point_frame is not None:
            status_text += f" | Point: ({clicked_point_frame[0]}, {clicked_point_frame[1]})"

        cv2.putText(
            display_frame,
            status_text,
            (40, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Display frame immediately in the OpenCV window
        cv2.imshow(window_name, display_frame)

        # Continuous waitKey to refresh the GUI and handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("\n[+] Exiting stream loop...")
            break
        elif key == ord("c") or key == ord("C"):
            clicked_point_frame = None
            clicked_point_display = None
            print("[+] Tracked coordinates cleared.")

    # Clean release of the persistent connection
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Stream closed cleanly.")


if __name__ == "__main__":
    run_camera()