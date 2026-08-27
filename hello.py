import os
import cv2

# Force reliable TCP transport for RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Camera RTSP URL for 10.27.1.77
rtsp_url = "rtsp://admin:admin123@10.27.1.77:554/Streaming/Channels/102/"

print(f"Connecting to RTSP stream at: {rtsp_url} ...")
print("Please wait...")

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Could not connect to camera RTSP stream.")
    exit(1)

print("✅ Camera connected successfully. Reading live stream...")
print("Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    cv2.imshow("IP Camera - 10.27.1.77", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Closing stream...")
        break

cap.release()
cv2.destroyAllWindows()
