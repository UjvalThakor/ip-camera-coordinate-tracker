# IP Camera Coordinate Tracker & Live RTSP Streamer

A high-performance Python application for continuous live streaming, real-time FPS monitoring, and interactive coordinate tracking from IP cameras (Hikvision, Dahua, ONVIF) over RTSP using OpenCV and FFmpeg TCP transport.

## Features

- 🚀 **Persistent Live Streaming**: Continuous single-connection RTSP video streaming with zero buffer lag.
- 📍 **Interactive Mouse Coordinate Tracking**: Click anywhere on the live video to detect and map original camera frame X/Y coordinates accurately.
- ⚡ **Dynamic Real-Time FPS**: Accurately computes true frame rate from incoming video frames.
- 📐 **Dynamic Resolution Detection**: Automatically detects and adapts to native camera frame dimensions (1080p, 720p, 480p, 2K/4K).
- 🛡️ **Corrupted Frame Protection**: Gracefully skips incomplete or dropped network frames without interrupting the stream.
- 🔄 **Safe TCP Transport**: Uses optimized FFmpeg TCP options to prevent packet drops and buffer overflows.

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

## Usage

Run the main application:
```bash
python caw.py
```

### Controls:
- **Left Click**: Place a crosshair and track the original `(X, Y)` frame coordinates.
- **`c`**: Clear tracked coordinates.
- **`q`** or **`ESC`**: Quit and close stream cleanly.

## Configuration

Edit `caw.py` to customize the camera RTSP URL and credentials:
```python
RTSP_URL = "rtsp://admin:admin123@<CAMERA_IP>:554/Streaming/Channels/102/"
```
