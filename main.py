import sys
import time
import math
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
import pyautogui  # <-- NEW: Added PyAutoGUI
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow
)

# Safety feature: Prevents the script from crashing if the mouse hits the edge of the screen
pyautogui.FAILSAFE = False

# -------- Auto-Download the MediaPipe Model --------
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Hand Landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

# -------- Video thread: capture and hand detection (Modern API) --------
class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cap = None
        
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.4
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.last_click_time = 0
        self.smoothing = 0.7  # Increased smoothing for a steadier system mouse
        self.pinch_threshold = 0.05
        self.click_cooldown = 0.5
        self.last_index = None
        self.last_thumb = None
        self.timestamp_ms = 0

    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.running = True

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)  # mirror
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                rgb = np.ascontiguousarray(rgb)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                index_coords = None
                thumb_coords = None

                try:
                    self.timestamp_ms += 33 
                    detection_result = self.detector.detect_for_video(mp_image, self.timestamp_ms)
                    
                    if detection_result.hand_landmarks:
                        hand = detection_result.hand_landmarks[0]
                        lm_index = hand[8]
                        lm_thumb = hand[4]

                        index_coords = (lm_index.x, lm_index.y)
                        thumb_coords = (lm_thumb.x, lm_thumb.y)

                        for lm in hand:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                        x1, y1 = int(lm_index.x * w), int(lm_index.y * h)
                        x2, y2 = int(lm_thumb.x * w), int(lm_thumb.y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 50), 2)
                except Exception as e:
                    pass

                display = frame.copy()
                if index_coords: self.last_index = index_coords
                if thumb_coords: self.last_thumb = thumb_coords
                
                self.frame_ready.emit(display)

        finally:
            if self.cap:
                self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


# -------- Main Window / UI --------
class StylishWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.animation_phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(40)
        self.multi_count = 5
        self.cursor_pos = [(0.5, 0.5) for _ in range(self.multi_count)]
        self.cursor_lerp = [0.75, 0.62, 0.5, 0.38, 0.28]
        self.target_x = 0.5
        self.target_y = 0.5
        self.cursor_clicked = False
        self.trails = [[] for _ in range(self.multi_count)]

    def step(self):
        self.animation_phase += 0.06
        for i in range(self.multi_count):
            cx, cy = self.cursor_pos[i]
            lerp = self.cursor_lerp[i]
            nx = cx + (self.target_x - cx) * lerp
            ny = cy + (self.target_y - cy) * lerp
            self.cursor_pos[i] = (nx, ny)
            t = self.trails[i]
            t.insert(0, (nx, ny))
            if len(t) > 10:
                t.pop()
        self.update()

    def set_cursor(self, nx: float, ny: float, clicked: bool):
        self.target_x = max(0.0, min(1.0, nx))
        self.target_y = max(0.0, min(1.0, ny))
        self.cursor_clicked = bool(clicked)
        if self.cursor_pos:
            self.cursor_pos[0] = (self.target_x, self.target_y)
            self.trails[0].insert(0, (self.target_x, self.target_y))
            if len(self.trails[0]) > 10:
                self.trails[0].pop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(12, 12, 12))
        w, h = self.width(), self.height()
        colors = [QColor(102, 178, 255), QColor(255, 153, 204), QColor(102, 255, 178), QColor(200, 180, 255)]
        n = 5
        base_margin = 24
        for i in range(n):
            phase = math.sin(self.animation_phase + i * 0.6)
            margin = base_margin + int(8 * i + 12 * phase)
            rect = (margin, margin + i * 4, w - 2 * margin, h - 2 * margin - i * 4)
            painter.setPen(colors[i % len(colors)])
            painter.drawRect(rect[0], rect[1], rect[2], rect[3])

        size = min(w, h) // 10
        palette = [QColor(102, 178, 255), QColor(255, 153, 204), QColor(102, 255, 178), QColor(200, 180, 255), QColor(180, 255, 220)]
        painter.setPen(Qt.NoPen)

        for j in range(self.multi_count):
            trail = self.trails[j]
            for k, (tx, ty) in enumerate(trail):
                px, py = int(tx * w), int(ty * h)
                col = palette[j % len(palette)]
                col.setAlpha(max(20, 200 - k * 22))
                painter.setBrush(col)
                radius = max(3, int(size * (0.6 - k * 0.04)))
                painter.drawEllipse(px - radius // 2, py - radius // 2, radius, radius)

        for j in range(self.multi_count):
            pxn, pyn = self.cursor_pos[j]
            cx, cy = int(pxn * w), int(pyn * h)
            col = palette[j % len(palette)]
            col.setAlpha(220 - j * 26)
            painter.setBrush(col)
            scale = 1.0 - 0.06 * j
            p1 = QPointF(cx - int(size * scale), cy - int(size * scale / 2))
            p2 = QPointF(cx - int(size * scale / 3), cy + int(size * scale / 10))
            p3 = QPointF(cx - int(size * scale / 2), cy + int(size * scale / 2))
            painter.drawPolygon(p1, p2, p3)

        if self.cursor_clicked:
            avg_x = sum(p[0] for p in self.cursor_pos) / len(self.cursor_pos)
            avg_y = sum(p[1] for p in self.cursor_pos) / len(self.cursor_pos)
            acx, acy = int(avg_x * w), int(avg_y * h)
            r = int(size * 2.6 * (0.6 + 0.4 * math.sin(self.animation_phase * 6)))
            painter.setPen(QColor(255, 215, 120, 160))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(acx - r // 2, acy - r // 2, r, r)
        painter.end()


class MouseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Mouse")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.stylish = StylishWidget()
        self.stylish.setMinimumSize(360, 360)
        layout.addWidget(self.stylish)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Control — Stylish Controller")
        self.setGeometry(120, 120, 900, 700)
        self.setStyleSheet("background-color: #0d0d0d; color: white;")
        central = QWidget()
        self.setCentralWidget(central)
        self.vbox = QVBoxLayout(central)
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setFixedHeight(640)
        self.video_label.setStyleSheet("background-color: black; border-radius:6px;")
        self.vbox.addWidget(self.video_label)

        # <-- NEW: Get screen dimensions for pyautogui
        self.screen_w, self.screen_h = pyautogui.size()

        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.on_frame)
        self.video_thread.start()

        self.virtual_x, self.virtual_y = 0.5, 0.5
        self.virtual_click = False
        self._click_viz_until = 0.0

        self.mouse_window = MouseWindow()
        self.mouse_window.resize(640, 640)
        self.mouse_window.show()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_hand_and_update_virtual_cursor)
        self.poll_timer.start(16)
        self.last_click_time = 0

    def closeEvent(self, event):
        self.video_thread.stop()
        try: self.mouse_window.close()
        except: pass
        event.accept()

    def on_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = qimg.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def poll_hand_and_update_virtual_cursor(self):
        idx = getattr(self.video_thread, 'last_index', None)
        th = getattr(self.video_thread, 'last_thumb', None)
        if idx is None: return

        nx, ny = max(0.0, min(1.0, idx[0])), max(0.0, min(1.0, idx[1]))
        s = self.video_thread.smoothing
        self.virtual_x = self.virtual_x + (nx - self.virtual_x) * (1 - s)
        self.virtual_y = self.virtual_y + (ny - self.virtual_y) * (1 - s)

        # <-- NEW: Move the actual Windows mouse cursor
        try:
            sys_x = int(self.virtual_x * self.screen_w)
            sys_y = int(self.virtual_y * self.screen_h)
            pyautogui.moveTo(sys_x, sys_y)
        except Exception as e:
            pass

        if th is not None:
            if math.hypot(nx - th[0], ny - th[1]) < self.video_thread.pinch_threshold:
                now = time.time()
                if now - self.last_click_time > self.video_thread.click_cooldown:
                    self.virtual_click = True
                    self._click_viz_until = now + 0.22
                    self.last_click_time = now
                    
                    # <-- NEW: Perform the actual Windows mouse click
                    try:
                        pyautogui.click()
                    except Exception as e:
                        pass

        if self.virtual_click and time.time() > self._click_viz_until:
            self.virtual_click = False

        try: self.mouse_window.stylish.set_cursor(self.virtual_x, self.virtual_y, self.virtual_click)
        except: pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())