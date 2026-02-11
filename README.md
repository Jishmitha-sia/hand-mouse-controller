# Hand-Mouse Controller v2.0

A computer vision application that turns your webcam into a touchless mouse. By utilizing the modern **MediaPipe Tasks API** for robust hand tracking and **PyAutoGUI** for system-level control, this app allows you to move your Windows cursor and click on items using simple hand gestures.

Budget Tony Stark interface: aggressively pinch the air to click things on your screen.

## Features
* **System-Wide Mouse Control**: Directly controls your actual operating system's mouse cursor.
* **Touchless Gestures**: 
  * Move your index finger to steer the mouse.
  * Pinch your thumb and index finger together to trigger a left-click.
* **Modern MediaPipe Backend**: Uses the `vision.RunningMode.VIDEO` API for stable, memory-efficient tracking without crashes.
* **Auto-Downloading Model**: Automatically fetches the required `hand_landmarker.task` machine learning model on its first run.
* **Stylized UI**: Includes a custom PyQt5 "Virtual Mouse" window with animated multi-cursor trails and click-ripple effects.

## Requirements
* **OS**: Windows 10 / 11
* **Python**: Python 3.10 or 3.11 (Recommended for MediaPipe stability)
* **Webcam**: Built-in or USB camera

## Installation

1. **Clone the repository or download the source code:**
   ```powershell
   git clone [https://github.com/your-username/hand-mouse-controller.git](https://github.com/your-username/hand-mouse-controller.git)
   cd hand-mouse-controller
   ```

2. **Set up a virtual environment (Recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
   *(Ensure `pyautogui` is installed successfully alongside OpenCV, MediaPipe, and PyQt5).*

## Usage

Run the main application script from your terminal:
```powershell
python main.py
```

1. **Initialization**: On the first run, the script will automatically download the ~3MB MediaPipe model file.
2. **Control**: Once the camera window opens, raise your hand into the frame.
3. **Move**: Point your index finger to move the cursor across your screen.
4. **Click**: Bring the tips of your thumb and index finger together (a pinch motion) to click.

### ⚠️ Emergency Stop (Failsafe)
Because this app takes over your system mouse, it can be tricky to close if your hand accidentally moves the cursor away from the exit button.
* **To instantly stop the mouse:** Hide your hand from the webcam's view. The mouse will release immediately.
* **To kill the program:** Use your physical hardware mouse to click back into your terminal window and press `Ctrl + C`.

## Packaging to `.exe`
You can build this project into a standalone executable so it can run without a terminal or Python installation.

1. Install PyInstaller:
   ```powershell
   pip install pyinstaller
   ```
2. Build the app:
   ```powershell
   pyinstaller --noconfirm --onefile --windowed main.py --name HandMouseController
   ```
3. Your executable will be generated inside the `dist/` folder.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.